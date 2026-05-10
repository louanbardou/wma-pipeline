#!/usr/bin/env python3
"""
wma_gradcam.py
==============
Grad-CAM heatmap generation and visualization for WMA detection.

Generates per-subject 3D heatmaps showing regions the model uses for WMA prediction.
Saves heatmaps as NIfTI files and produces aggregate visualizations.

Usage:
    python wma_gradcam.py \
        --checkpoint runs/wma/best_model.pt \
        --manifest data/manifest.csv \
        --fold 0 \
        --out_dir heatmaps/ \
        --top_k 50
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wma_pipeline
wma_pipeline.init_ml()
from wma_pipeline import (
    Config,
    WMAClassifier,
    WMADataset,
    get_fold_splits,
    get_val_transforms,
    get_synthetic_transforms,
    set_seed,
)


# ============================================================
# SECTION 1: GRAD-CAM ENGINE
# ============================================================

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D volumes.

    Registers forward/backward hooks on a target layer to capture
    activations and gradients, then computes the weighted activation map.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # Swin transformer outputs (B, T, C) tokens; reshape to (B, C, D, H, W)
        if output.dim() == 3:
            B, T, C = output.shape
            s = round(T ** (1/3))
            output = output.permute(0, 2, 1).reshape(B, C, s, s, s)
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        if grad.dim() == 3:
            B, T, C = grad.shape
            s = round(T ** (1/3))
            grad = grad.permute(0, 2, 1).reshape(B, C, s, s, s)
        self.gradients = grad.detach()

    def __call__(self, x, target_size=None):
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        x : torch.Tensor (B, C, D, H, W)
        target_size : tuple, optional
            Spatial size to upsample the heatmap to. Defaults to input spatial dims.

        Returns
        -------
        heatmap : torch.Tensor (B, D, H, W) in [0, 1]
        logits : torch.Tensor (B,)
        """
        self.model.eval()
        x.requires_grad_(True)

        logits = self.model(x).squeeze(1)

        self.model.zero_grad()
        logits.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        # Per-channel importance weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, d, h, w)
        cam = F.relu(cam)  # only positive contributions

        # Upsample to target size
        if target_size is None:
            target_size = x.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)

        # Normalize per-sample to [0, 1]
        B = cam.shape[0]
        cam = cam.squeeze(1)  # (B, D, H, W)
        for i in range(B):
            c = cam[i]
            cmin, cmax = c.min(), c.max()
            if cmax > cmin:
                cam[i] = (c - cmin) / (cmax - cmin)

        return cam.detach(), logits.detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ============================================================
# SECTION 2: HEATMAP GENERATION
# ============================================================

def generate_heatmaps(args):
    """Generate per-subject GradCAM heatmaps and save as NIfTI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})

    model = WMAClassifier(
        backbone=saved_cfg.get("backbone", "resnet"),
        in_channels=saved_cfg.get("in_channels", 3),
        dropout=saved_cfg.get("dropout", 0.4),
    ).to(device)

    state_dict = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # GradCAM
    target_layer = model.get_cam_target()
    gradcam = GradCAM3D(model, target_layer)

    # Data
    import pandas as pd
    set_seed(42)
    df = pd.read_csv(args.manifest)
    _, val_df = get_fold_splits(df, 5, args.fold, 42)

    # Filter to positives only (or top_k by prediction score)
    synthetic = getattr(args, "synthetic", False)
    transform = get_synthetic_transforms(size=32) if synthetic else get_val_transforms()
    val_ds = WMADataset(val_df.reset_index(drop=True), transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0 if synthetic else 4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating GradCAM heatmaps for {len(val_ds)} validation samples...")

    all_cams_pos = []
    results = []

    for i, (image, label, _) in enumerate(val_loader):
        image = image.to(device)
        cam, logits = gradcam(image, target_size=image.shape[2:])

        prob = torch.sigmoid(logits).item()
        label_val = int(label.item())
        row = val_df.iloc[i]
        sid = row["subject_id"]
        ses = row["session"]

        results.append({
            "subject_id": sid, "session": ses,
            "label": label_val, "prob": prob,
        })

        # Save heatmap for positives and high-confidence predictions
        if label_val == 1 or prob > 0.5:
            cam_np = cam[0].cpu().numpy()

            # Create NIfTI with identity affine (same space as cropped input)
            nii = nib.Nifti1Image(cam_np.astype(np.float32), affine=np.eye(4))
            fname = out_dir / f"{sid}_{ses}_gradcam.nii.gz"
            nib.save(nii, str(fname))

            if label_val == 1:
                all_cams_pos.append(cam_np)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_ds)}")

    gradcam.remove_hooks()

    # --- Aggregate heatmap for true positives ---
    if all_cams_pos:
        mean_cam = np.mean(all_cams_pos, axis=0)
        nii = nib.Nifti1Image(mean_cam.astype(np.float32), affine=np.eye(4))
        nib.save(nii, str(out_dir / "aggregate_gradcam_positives.nii.gz"))
        print(f"\nAggregate heatmap saved ({len(all_cams_pos)} positive subjects)")

    # Save results table
    import pandas as pd
    pd.DataFrame(results).to_csv(out_dir / "gradcam_results.csv", index=False)
    print(f"Results table saved -> {out_dir / 'gradcam_results.csv'}")

    # --- Visualization ---
    _plot_aggregate(all_cams_pos, out_dir)
    _plot_top_cases(results, out_dir, all_cams_pos)

    print(f"\nDone. Heatmaps in {out_dir}")


# ============================================================
# SECTION 3: VISUALIZATION
# ============================================================

def _plot_aggregate(cams_pos, out_dir):
    """Plot axial/coronal/sagittal slices of aggregate heatmap."""
    if not cams_pos:
        return

    mean_cam = np.mean(cams_pos, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Aggregate GradCAM — {len(cams_pos)} WMA+ subjects", fontsize=14)

    titles = ["Axial (z-mid)", "Coronal (y-mid)", "Sagittal (x-mid)"]
    slices = [
        mean_cam[mean_cam.shape[0]//2, :, :],
        mean_cam[:, mean_cam.shape[1]//2, :],
        mean_cam[:, :, mean_cam.shape[2]//2],
    ]

    for ax, s, t in zip(axes, slices, titles):
        im = ax.imshow(s.T, cmap="hot", origin="lower", vmin=0, vmax=s.max())
        ax.set_title(t)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Aggregate plot saved -> {out_dir / 'aggregate_gradcam.png'}")


def _plot_top_cases(results, out_dir, cams_pos):
    """Plot mosaic of highest-confidence WMA+ and WMA- predictions."""
    import pandas as pd
    df = pd.DataFrame(results)

    # Top false positives (high prob, label=0) and false negatives (low prob, label=1)
    fp = df[(df["label"] == 0)].nlargest(5, "prob")
    fn = df[(df["label"] == 1)].nsmallest(5, "prob")

    summary = "Top false positives (predicted WMA but label=0):\n"
    for _, r in fp.iterrows():
        summary += f"  {r['subject_id']} {r['session']} prob={r['prob']:.4f}\n"
    summary += "\nTop false negatives (missed WMA, label=1):\n"
    for _, r in fn.iterrows():
        summary += f"  {r['subject_id']} {r['session']} prob={r['prob']:.4f}\n"

    with open(out_dir / "error_analysis.txt", "w") as f:
        f.write(summary)
    print(summary)


# ============================================================
# SECTION 4: MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GradCAM heatmaps for WMA detection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out_dir", default="heatmaps/")
    parser.add_argument("--top_k", type=int, default=50, help="Max heatmaps to save")
    parser.add_argument("--synthetic", action="store_true", help="Use random tensors")
    args = parser.parse_args()
    generate_heatmaps(args)


if __name__ == "__main__":
    main()
