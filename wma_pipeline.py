#!/usr/bin/env python3
"""
wma_pipeline.py
===============
Single-file pipeline for White Matter Abnormality (WMA) detection from T1+T2 MRI.
Binary classification on the ABCD dataset (~500 WMA+ subjects, ~6000 WMA- subjects).

Subcommands:
    labels   — Fuse two CSV sources into binary WMA labels
    manifest — Match labels to NIfTI files on disk
    train    — 2-stage training: CE+Focal warmup → APLoss+SOAP fine-tune
    eval     — Evaluate checkpoint with TTA, optimal threshold, per-site metrics

Key design choices (80/20 — max AUPREC impact):
    - ResNet-3D ~23M params (vs 62M Swin UNETR) — better param/positive ratio
    - 3-channel input: T1 + T2 + T2/T1 ratio (compensates for missing FLAIR)
    - 2-stage loss: CE warmup gives APLoss a good starting score distribution
    - TensorBoard offline (works on isolated HPC, no internet needed)
    - bf16 mixed precision + gradient accumulation + EMA
"""

# ============================================================
# SECTION 0: IMPORTS & CONSTANTS
# ============================================================

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PATCH_SIZE = (128, 128, 128)
TARGET_VOXEL = (1.0, 1.0, 1.0)

EVENT_MAP = {
    "baseline_year_1_arm_1": "ses-00A",
    "1_year_follow_up_y_arm_1": "ses-01A",
    "2_year_follow_up_y_arm_1": "ses-02A",
    "3_year_follow_up_y_arm_1": "ses-03A",
    "4_year_follow_up_y_arm_1": "ses-04A",
    "6_year_follow_up_y_arm_1": "ses-06A",
    "8_year_follow_up_y_arm_1": "ses-08A",
}


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

@dataclass
class Config:
    manifest: str = "data/manifest.csv"
    out_dir: str = "runs/wma"
    cache_dir: str = ""
    backbone: str = "resnet"
    in_channels: int = 3
    dropout: float = 0.4
    drop_path: float = 0.1
    epochs: int = 40
    warmup_epochs: int = 15
    batch_size: int = 4
    effective_batch_size: int = 16
    lr: float = 3e-4
    lr_backbone_factor: float = 0.1
    freeze_epochs: int = 5
    weight_decay: float = 5e-4
    aploss_gamma: float = 0.9
    epoch_decay: float = 1e-3
    seed: int = 42
    fold: int = 0
    n_folds: int = 5
    use_ema: bool = True
    ema_decay: float = 0.9995
    use_bf16: bool = True
    patience: int = 10
    use_tta: bool = False
    checkpoint: str = ""
    pretrained_weights: str = ""


# ============================================================
# SECTION 2: LABEL FUSION & MANIFEST BUILDER
# ============================================================

def cmd_labels(args):
    """Fuse two CSV sources into binary WMA labels."""
    csv1 = pd.read_csv(args.csv1)
    csv2 = pd.read_csv(args.csv2)

    csv1 = csv1.rename(columns={"id_redcap": "subject_id", "redcap_event_name": "event_name"})
    csv1["subject_id"] = csv1["subject_id"].astype(str).str.strip()
    csv1["event_name"] = csv1["event_name"].astype(str).str.strip()
    csv1["session"] = csv1["event_name"].map(EVENT_MAP)
    csv1["has_wma"] = (
        csv1["Predicted Findings"].fillna("")
        .str.contains("White Matter Abnormality", case=False, na=False)
        .astype(int)
    )

    csv2["subject_id"] = csv2["subject_id"].astype(str).str.strip()
    csv2["event_name"] = csv2["event_name"].astype(str).str.strip()
    csv2["session"] = csv2["event_name"].map(EVENT_MAP)
    csv2["has_wma"] = (
        csv2["finding"].fillna("")
        .str.contains("White Matter Abnormality", case=False, na=False)
        .astype(int)
    )

    agg1 = (csv1.dropna(subset=["session"])
            .groupby(["subject_id", "session"])["has_wma"].max()
            .reset_index().rename(columns={"has_wma": "wma_csv1"}))
    agg2 = (csv2.dropna(subset=["session"])
            .groupby(["subject_id", "session"])["has_wma"].max()
            .reset_index().rename(columns={"has_wma": "wma_csv2"}))

    merged = pd.merge(agg1, agg2, on=["subject_id", "session"], how="outer")
    merged["wma_csv1"] = merged["wma_csv1"].fillna(0).astype(int)
    merged["wma_csv2"] = merged["wma_csv2"].fillna(0).astype(int)
    merged["label"] = (merged["wma_csv1"] | merged["wma_csv2"]).astype(int)

    result = merged[["subject_id", "session", "label"]].sort_values(
        ["subject_id", "session"]).reset_index(drop=True)

    n_total = len(result)
    n_pos = int(result["label"].sum())
    n_subj = result["subject_id"].nunique()
    n_subj_pos = result[result["label"] == 1]["subject_id"].nunique()

    print(f"Label fusion complete:")
    print(f"  Total scan-sessions  : {n_total}")
    print(f"  WMA+ (label=1)       : {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"  WMA- (label=0)       : {n_total - n_pos}")
    print(f"  Unique subjects      : {n_subj}")
    print(f"  Unique WMA+ subjects : {n_subj_pos}")
    print(f"\n  Per session:")
    for ses, grp in result.groupby("session"):
        print(f"    {ses}  total={len(grp):>5}  WMA+={int(grp['label'].sum()):>4}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


def cmd_manifest(args):
    """Match labels to NIfTI files on disk."""
    labels = pd.read_csv(args.labels)
    data_root = Path(args.data_root)

    label_map = {}
    for _, row in labels.iterrows():
        sid = str(row["subject_id"]).strip()
        ses = str(row["session"]).strip()
        label_map[(sid, ses)] = int(row["label"])

    rows = []
    n_scanned = n_both = n_labeled = 0

    for sub_dir in sorted(data_root.glob("sub-*")):
        bare_id = sub_dir.name[4:]
        ndar_id = f"NDAR_INV{bare_id}"
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            anat = ses_dir / "anat"
            if not anat.exists():
                continue
            n_scanned += 1
            t1_files = sorted(anat.glob("*T1w.nii.gz"))
            t2_files = sorted(anat.glob("*T2w.nii.gz"))
            if not t1_files or not t2_files:
                continue
            n_both += 1
            # Try all ID formats — can't use `or` because label=0 is falsy
            label = label_map.get((ndar_id, ses))
            if label is None:
                label = label_map.get((bare_id, ses))
            if label is None:
                label = label_map.get((f"sub-{bare_id}", ses))
            if label is None:
                continue
            n_labeled += 1
            rows.append({
                "subject_id": f"sub-{bare_id}", "session": ses,
                "t1w_path": str(t1_files[0]), "t2w_path": str(t2_files[0]),
                "label": int(label),
            })

    df = pd.DataFrame(rows).sort_values(["subject_id", "session"]).reset_index(drop=True)
    print(f"Manifest: {n_scanned} scanned, {n_both} with T1+T2, {n_labeled} labeled")
    print(f"  Total rows : {len(df)},  WMA+={int(df['label'].sum())},  subjects={df['subject_id'].nunique()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")


# ============================================================
# SECTIONS 3-8: ML CODE (loaded on demand via init_ml())
# ============================================================

def init_ml():
    """Import heavy ML libraries and define model/dataset/transform classes.
    Call this before using any ML functionality. Safe to call multiple times."""
    if hasattr(init_ml, "_done"):
        return
    init_ml._done = True

    import torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
    from sklearn.metrics import (
        average_precision_score as _aps, brier_score_loss as _bsl,
        precision_recall_curve as _prc, roc_auc_score as _ras, roc_curve as _rc,
    )
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF
    from monai.networks.nets import SwinUNETR as _SwinUNETR, resnet10 as _resnet10
    from monai.transforms import (
        ConcatItemsd, Compose, CropForegroundd, DeleteItemsd,
        EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Orientationd,
        RandAffined, RandCoarseDropoutd, RandFlipd, RandGaussianNoised,
        RandGaussianSmoothd, RandRotate90d, RandScaleIntensityd,
        RandSpatialCropd, Spacingd, SpatialPadd, ToTensord,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter as _SW
    except ImportError:
        _SW = None

    # ── Publish to module namespace ──
    _m = sys.modules[__name__]
    _m.torch = torch
    _m.nn = _nn
    _m.F = _F
    _m.DataLoader = _DataLoader
    _m.SummaryWriter = _SW
    _m.average_precision_score = _aps
    _m.brier_score_loss = _bsl
    _m.precision_recall_curve = _prc
    _m.roc_auc_score = _ras
    _m.roc_curve = _rc
    _m.StratifiedGroupKFold = _SGKF

    # ── Transforms ──

    def _compute_ratio(data):
        t1, t2 = data["t1w"], data["t2w"]
        eps = 1e-3
        ratio = t2 / (t1.abs() + eps)
        ratio = ratio.clamp(0, 5)
        mask = t1.abs() > 0.01
        if mask.any():
            vals = ratio[mask]
            ratio = (ratio - vals.mean()) / (vals.std() + eps)
        data["ratio"] = ratio
        return data

    def get_train_transforms():
        return Compose([
            LoadImaged(keys=["t1w", "t2w"]),
            EnsureChannelFirstd(keys=["t1w", "t2w"]),
            Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
            Spacingd(keys=["t1w", "t2w"], pixdim=TARGET_VOXEL, mode="bilinear"),
            CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),
            NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),
            RandSpatialCropd(keys=["t1w", "t2w"], roi_size=PATCH_SIZE, random_size=False),
            RandFlipd(keys=["t1w", "t2w"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["t1w", "t2w"], prob=0.3, max_k=3),
            RandAffined(keys=["t1w", "t2w"], prob=0.3,
                        rotate_range=(0.1, 0.1, 0.1), scale_range=(0.05, 0.05, 0.05),
                        translate_range=(5, 5, 5), mode="bilinear", padding_mode="border"),
            RandGaussianNoised(keys=["t1w", "t2w"], prob=0.2, std=0.05),
            RandGaussianSmoothd(keys=["t1w", "t2w"], prob=0.2,
                                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensityd(keys=["t1w", "t2w"], prob=0.3, factors=0.1),
            RandCoarseDropoutd(keys=["t1w", "t2w"], prob=0.2,
                               holes=3, spatial_size=(16, 16, 16), fill_value=0),
            _compute_ratio,
            ConcatItemsd(keys=["t1w", "t2w", "ratio"], name="image"),
            DeleteItemsd(keys=["t1w", "t2w", "ratio"]),
            ToTensord(keys=["image"]),
        ])

    def get_val_transforms():
        return Compose([
            LoadImaged(keys=["t1w", "t2w"]),
            EnsureChannelFirstd(keys=["t1w", "t2w"]),
            Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
            Spacingd(keys=["t1w", "t2w"], pixdim=TARGET_VOXEL, mode="bilinear"),
            CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),
            NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),
            RandSpatialCropd(keys=["t1w", "t2w"], roi_size=PATCH_SIZE,
                             random_size=False, random_center=False),
            _compute_ratio,
            ConcatItemsd(keys=["t1w", "t2w", "ratio"], name="image"),
            DeleteItemsd(keys=["t1w", "t2w", "ratio"]),
            ToTensord(keys=["image"]),
        ])

    _m.get_train_transforms = get_train_transforms
    _m.get_val_transforms = get_val_transforms

    # ── Synthetic transforms for dry-run testing ──

    def get_synthetic_transforms(size=32):
        """Return a transform that generates random 3-channel tensors (no NIfTI needed).
        Uses small volumes for fast CPU testing."""
        def _synth_transform(data):
            return {"image": torch.randn(3, size, size, size)}
        return _synth_transform

    _m.get_synthetic_transforms = get_synthetic_transforms

    # ── Dataset ──

    class WMADataset(_Dataset):
        def __init__(self, df, transform, cache_dir=None):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            self.targets = self.df["label"].astype(int).tolist()
            self.cache_dir = Path(cache_dir) if cache_dir else None
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            label = torch.tensor(float(row["label"]), dtype=torch.float32)
            if self.cache_dir is not None:
                key = f"{row['subject_id']}_{row['session']}.pt"
                cache_path = self.cache_dir / key
                if cache_path.exists():
                    try:
                        return torch.load(cache_path, weights_only=False), label, idx
                    except Exception:
                        cache_path.unlink(missing_ok=True)
                data = self.transform({"t1w": row["t1w_path"], "t2w": row["t2w_path"]})
                image = data["image"]
                torch.save(image, cache_path)
                return image, label, idx
            data = self.transform({"t1w": row.get("t1w_path", ""), "t2w": row.get("t2w_path", "")})
            if isinstance(data, dict):
                return data["image"], label, idx
            return data, label, idx

    _m.WMADataset = WMADataset

    # ── Model ──

    class WMAClassifier(_nn.Module):
        """
        WMA binary classifier.
        - "resnet": ResNet-10 3D (~14.4M params, hidden_dim=512)
        - "swin_tiny": SwinUNETR feature_size=24 (~15.8M params, hidden_dim=384)
        """
        def __init__(self, backbone="resnet", in_channels=3, dropout=0.4, drop_path=0.1):
            super().__init__()
            self.backbone_name = backbone
            if backbone == "resnet":
                self.encoder = _resnet10(
                    spatial_dims=3, n_input_channels=in_channels, num_classes=1,
                )
                hidden_dim = 512  # ResNet-10 final layer = 512 channels
                self.encoder.fc = _nn.Identity()
                self.gap = _nn.AdaptiveAvgPool3d(1)
                # Stochastic depth: dropout on feature maps between stages
                self.drop_path = _nn.Dropout3d(p=drop_path) if drop_path > 0 else _nn.Identity()
            elif backbone == "swin_tiny":
                self.encoder = _SwinUNETR(
                    in_channels=in_channels, out_channels=2,
                    feature_size=24, use_checkpoint=True,
                    spatial_dims=3, drop_rate=drop_path,
                )
                hidden_dim = 384  # feature_size * 16
                self.gap = _nn.AdaptiveAvgPool3d(1)
                self.drop_path = _nn.Identity()
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
            self.head = _nn.Sequential(
                _nn.Flatten(), _nn.LayerNorm(hidden_dim), _nn.Dropout(dropout),
                _nn.Linear(hidden_dim, 128), _nn.GELU(), _nn.Dropout(dropout / 2),
                _nn.Linear(128, 1),
            )

        def forward_features(self, x):
            if self.backbone_name == "resnet":
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.act(x)
                x = self.encoder.maxpool(x)
                x = self.encoder.layer1(x)
                x = self.drop_path(x)
                x = self.encoder.layer2(x)
                x = self.drop_path(x)
                x = self.encoder.layer3(x)
                x = self.drop_path(x)
                x = self.encoder.layer4(x)
                return x
            else:
                return self.encoder.swinViT(x, normalize=True)[-1]

        def forward(self, x):
            return self.head(self.gap(self.forward_features(x)))

        def get_cam_target(self):
            if self.backbone_name == "resnet":
                return self.encoder.layer4[-1]
            return self.encoder.swinViT.layers4[-1]

        def get_layer_groups(self):
            """Return param groups for layer-wise LR decay."""
            if self.backbone_name == "resnet":
                return [
                    list(self.encoder.conv1.parameters()) + list(self.encoder.bn1.parameters()),
                    list(self.encoder.layer1.parameters()),
                    list(self.encoder.layer2.parameters()),
                    list(self.encoder.layer3.parameters()),
                    list(self.encoder.layer4.parameters()),
                    list(self.head.parameters()),
                ]
            else:
                return [
                    list(self.encoder.parameters()),
                    list(self.head.parameters()),
                ]

    _m.WMAClassifier = WMAClassifier

    # ── Training utilities ──

    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.001):
            self.patience, self.min_delta = patience, min_delta
            self.counter, self.best_score = 0, None
        def __call__(self, score):
            if self.best_score is None or score > self.best_score + self.min_delta:
                self.best_score, self.counter = score, 0
                return False
            self.counter += 1
            return self.counter >= self.patience

    class FocalBCELoss(_nn.Module):
        def __init__(self, gamma=2.0, alpha=0.85, focal_weight=0.5):
            super().__init__()
            self.gamma, self.alpha, self.focal_weight = gamma, alpha, focal_weight
        def forward(self, logits, targets):
            bce = _F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            pt = targets * p + (1 - targets) * (1 - p)
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal = alpha_t * (1 - pt) ** self.gamma * bce
            return bce.mean() + self.focal_weight * focal.mean()

    _m.EarlyStopping = EarlyStopping
    _m.FocalBCELoss = FocalBCELoss

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def get_fold_splits(df, n_splits, fold, seed):
        sgkf = _SGKF(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (tr, va) in enumerate(sgkf.split(df, df["label"].values, df["subject_id"].values)):
            if i == fold:
                return df.iloc[tr], df.iloc[va]
        raise ValueError(f"Fold {fold} not found")

    _m.set_seed = set_seed
    _m.get_fold_splits = get_fold_splits


# ============================================================
# SECTION 7: TRAINING LOOP
# ============================================================

def _ml():
    """Return this module's namespace (where init_ml publishes symbols)."""
    return sys.modules[__name__]


def cmd_train(args):
    """Two-stage training: CE+Focal warmup -> APLoss+SOAP fine-tune."""
    init_ml()
    # Pull ML symbols into local scope for readability
    M = _ml()
    torch = M.torch; nn = M.nn; F = M.F
    DataLoader = M.DataLoader; SummaryWriter = M.SummaryWriter
    WMADataset = M.WMADataset; WMAClassifier = M.WMAClassifier
    FocalBCELoss = M.FocalBCELoss; EarlyStopping = M.EarlyStopping
    set_seed = M.set_seed; get_fold_splits = M.get_fold_splits
    average_precision_score = M.average_precision_score
    roc_auc_score = M.roc_auc_score
    get_train_transforms = M.get_train_transforms
    get_val_transforms = M.get_val_transforms
    get_synthetic_transforms = M.get_synthetic_transforms

    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    if getattr(args, "no_ema", False):
        cfg.use_ema = False

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    print(f"Device: {device}, AMP: {use_amp}")

    # TensorBoard
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs")) if SummaryWriter else None

    # Data
    df = pd.read_csv(cfg.manifest)
    n_pos = int(df["label"].sum())
    print(f"Manifest: {len(df)} rows, {n_pos} WMA+ ({n_pos/len(df)*100:.1f}%)")

    train_df, val_df = get_fold_splits(df, cfg.n_folds, cfg.fold, cfg.seed)
    print(f"Fold {cfg.fold}: train={len(train_df)} ({int(train_df['label'].sum())} pos), "
          f"val={len(val_df)} ({int(val_df['label'].sum())} pos)")

    # Detect synthetic mode (no NIfTI paths)
    synthetic = getattr(args, "synthetic", False)
    if synthetic:
        transform_train = get_synthetic_transforms(size=32)
        transform_val = get_synthetic_transforms(size=32)
    else:
        transform_train = get_train_transforms()
        transform_val = get_val_transforms()

    cache_dir = cfg.cache_dir or None
    train_ds = WMADataset(train_df, transform_train, cache_dir=None if synthetic else cache_dir)
    val_ds = WMADataset(val_df, transform_val)

    num_workers = 0 if synthetic else 8
    train_loader_s1 = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = WMAClassifier(
        backbone=cfg.backbone, in_channels=cfg.in_channels,
        dropout=cfg.dropout, drop_path=cfg.drop_path,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg.backbone}, {n_params:.1f}M params")

    if cfg.pretrained_weights:
        ckpt = torch.load(cfg.pretrained_weights, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"Pretrained: missing={len(miss)}, unexpected={len(unexp)}")

    # EMA
    ema_model = None
    if cfg.use_ema:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda a, n, _: cfg.ema_decay * a + (1 - cfg.ema_decay) * n)

    # Progressive unfreezing
    def set_backbone_grad(req):
        if cfg.backbone == "resnet":
            for name, p in model.encoder.named_parameters():
                if "fc" not in name:
                    p.requires_grad = req
        else:
            for p in model.encoder.parameters():
                p.requires_grad = req

    set_backbone_grad(False)
    print(f"Backbone frozen for epochs 1-{cfg.freeze_epochs}")

    accum_steps = max(1, cfg.effective_batch_size // cfg.batch_size)
    print(f"Grad accum: {accum_steps}x (effective bs={cfg.effective_batch_size})")

    # Stage 1 loss + optimizer
    loss_fn_s1 = FocalBCELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    history = []
    best_auprec = 0.0
    global_step = 0
    early_stop = EarlyStopping(patience=cfg.patience)
    stage = 1
    train_loader = train_loader_s1
    loss_fn_s2 = None

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone with layer-wise LR decay
        if epoch == cfg.freeze_epochs + 1 and stage == 1:
            set_backbone_grad(True)
            layer_groups = model.get_layer_groups()
            n_groups = len(layer_groups)
            # Layer-wise LR decay: deeper layers get lower LR (factor 0.75 per layer)
            lr_decay = 0.75
            param_groups = []
            for i, group_params in enumerate(layer_groups):
                group_lr = cfg.lr * cfg.lr_backbone_factor * (lr_decay ** (n_groups - 1 - i))
                if i == n_groups - 1:  # head gets full LR
                    group_lr = cfg.lr
                param_groups.append({"params": group_params, "lr": group_lr})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
            lrs = [f"{pg['lr']:.1e}" for pg in param_groups]
            print(f"Epoch {epoch}: backbone unfrozen, layer-wise LRs: {lrs}")

        # Switch to stage 2
        if epoch == cfg.warmup_epochs + 1 and stage == 1:
            stage = 2
            print(f"\n{'='*60}\nSTAGE 2: APLoss + SOAP (epoch {epoch})\n{'='*60}")
            try:
                from libauc.losses import APLoss
                from libauc.optimizers import SOAP
                from libauc.sampler import DualSampler
            except ImportError:
                print("WARNING: libauc not installed, continuing with CE loss for stage 2")
                APLoss = SOAP = DualSampler = None

            if APLoss is not None:
                loss_fn_s2 = APLoss(data_len=len(train_df), margin=1.0, gamma=cfg.aploss_gamma)
                # SOAP doesn't support param_groups dicts — pass flat param list
                lr_s2 = cfg.lr / 3
                optimizer = SOAP(
                    model.parameters(), lr=lr_s2,
                    epoch_decay=cfg.epoch_decay, weight_decay=cfg.weight_decay,
                )
                train_sampler = DualSampler(train_ds, batch_size=cfg.batch_size,
                                            num_pos=1, sampling_rate=None, random_seed=cfg.seed)
                train_loader = DataLoader(
                    train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=(device.type == "cuda"),
                    persistent_workers=(num_workers > 0),
                )
            early_stop = EarlyStopping(patience=cfg.patience)

        # Cosine LR
        if stage == 1:
            progress = min(epoch / cfg.warmup_epochs, 1.0)
        else:
            progress = min((epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs), 1.0)
        cos_factor = 0.5 * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]
            pg["lr"] = pg["initial_lr"] * max(cos_factor, 0.001)
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        model.train()
        epoch_loss, epoch_gnorm, n_batches = 0.0, 0.0, 0
        tr_logits_all, tr_labels_all = [], []
        optimizer.zero_grad()

        for step, (images, batch_labels, batch_idx) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_idx = batch_idx.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(images).squeeze(1)
            logits_f32 = logits.float()

            if stage == 2 and loss_fn_s2 is not None:
                loss = loss_fn_s2(logits_f32, batch_labels, batch_idx)
            else:
                loss = loss_fn_s1(logits_f32, batch_labels)

            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update_parameters(model)
                epoch_gnorm += gnorm.item()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            tr_logits_all.extend(logits_f32.detach().cpu().tolist())
            tr_labels_all.extend(batch_labels.cpu().tolist())

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        eval_m = ema_model if ema_model else model
        eval_m.eval()
        va_logits, va_labels = [], []
        with torch.no_grad():
            for images, batch_labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                    logits = eval_m(images).squeeze(1)
                va_logits.extend(logits.float().cpu().tolist())
                va_labels.extend(batch_labels.tolist())

        tr_probs = torch.sigmoid(torch.tensor(tr_logits_all)).numpy()
        va_probs = torch.sigmoid(torch.tensor(va_logits)).numpy()
        tr_auprec = average_precision_score(tr_labels_all, tr_probs)
        va_auprec = average_precision_score(va_labels, va_probs)
        va_auroc = roc_auc_score(va_labels, va_probs)
        elapsed = time.time() - t0

        print(f"[S{stage}] Epoch {epoch:3d}/{cfg.epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
              f"tr_AP={tr_auprec:.4f}  va_AP={va_auprec:.4f}  va_AUC={va_auroc:.4f}  {elapsed:.0f}s")

        if writer:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/auprec", tr_auprec, epoch)
            writer.add_scalar("val/auprec", va_auprec, epoch)
            writer.add_scalar("val/auroc", va_auroc, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)
            writer.add_scalar("train/stage", stage, epoch)
            if epoch % 5 == 0 or epoch == cfg.epochs:
                writer.add_pr_curve("val/pr_curve", np.array(va_labels), va_probs, epoch)

        history.append({"epoch": epoch, "stage": stage, "loss": avg_loss,
                        "train_auprec": tr_auprec, "val_auprec": va_auprec,
                        "val_auroc": va_auroc, "lr": current_lr})

        if va_auprec > best_auprec:
            best_auprec = va_auprec
            torch.save({
                "epoch": epoch, "stage": stage,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.module.state_dict() if ema_model else None,
                "val_auprec": va_auprec, "val_auroc": va_auroc, "config": vars(cfg),
            }, out_dir / "best_model.pt")
            print(f"  -> New best AUPREC={best_auprec:.4f}")

        if epoch > cfg.freeze_epochs and early_stop(va_auprec):
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch == cfg.warmup_epochs and va_auprec < 0.18:
            print(f"\n  WARNING: Stage 1 val AUPREC={va_auprec:.4f} < 0.18 — possible issue\n")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    if writer:
        writer.add_hparams(
            {"backbone": cfg.backbone, "lr": cfg.lr, "fold": cfg.fold},
            {"hparam/best_auprec": best_auprec},
        )
        writer.close()
    print(f"\nDone. Best AUPREC: {best_auprec:.4f} | Checkpoint: {out_dir / 'best_model.pt'}")


# ============================================================
# SECTION 8: EVALUATION
# ============================================================

def cmd_eval(args):
    """Evaluate a checkpoint with optional TTA."""
    init_ml()
    M = _ml()
    torch = M.torch; DataLoader = M.DataLoader
    WMADataset = M.WMADataset; WMAClassifier = M.WMAClassifier
    get_fold_splits = M.get_fold_splits; set_seed = M.set_seed
    average_precision_score = M.average_precision_score
    roc_auc_score = M.roc_auc_score
    brier_score_loss = M.brier_score_loss
    precision_recall_curve = M.precision_recall_curve
    roc_curve = M.roc_curve
    get_val_transforms = M.get_val_transforms
    get_synthetic_transforms = M.get_synthetic_transforms

    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_bf16 and device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", {})

    model = WMAClassifier(
        backbone=saved_cfg.get("backbone", cfg.backbone),
        in_channels=saved_cfg.get("in_channels", cfg.in_channels),
        dropout=saved_cfg.get("dropout", cfg.dropout),
    ).to(device)

    sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    df = pd.read_csv(cfg.manifest)
    _, val_df = get_fold_splits(df, cfg.n_folds, cfg.fold, cfg.seed)

    synthetic = getattr(args, "synthetic", False)
    transform = get_synthetic_transforms(size=32) if synthetic else get_val_transforms()
    val_ds = WMADataset(val_df, transform)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0 if synthetic else 8)

    print(f"Eval fold {cfg.fold}: {len(val_df)} samples ({int(val_df['label'].sum())} pos), TTA={cfg.use_tta}")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, batch_labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(images).squeeze(1)
                if cfg.use_tta:
                    tta = [logits]
                    for ax in [2, 3, 4]:
                        tta.append(model(torch.flip(images, dims=[ax])).squeeze(1))
                    logits = torch.stack(tta).mean(0)
            all_logits.extend(logits.float().cpu().tolist())
            all_labels.extend(batch_labels.tolist())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels_arr = np.array(all_labels)

    auprec = average_precision_score(labels_arr, probs)
    auroc = roc_auc_score(labels_arr, probs)
    brier = brier_score_loss(labels_arr, probs)

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels_arr, probs)
    f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    opt_idx = np.argmax(f1s)
    opt_thresh = thresh_arr[min(opt_idx, len(thresh_arr) - 1)]

    fpr, tpr, _ = roc_curve(labels_arr, probs)
    s90_idx = np.searchsorted(1 - fpr[::-1], 0.90)
    sens90 = tpr[::-1][min(s90_idx, len(tpr) - 1)] if s90_idx < len(tpr) else 0.0

    print(f"\n{'='*50}\n  AUPREC={auprec:.4f}  AUROC={auroc:.4f}  Brier={brier:.4f}")
    print(f"  Opt threshold={opt_thresh:.4f}  F1={f1s[opt_idx]:.4f}  Sens@Spec90={sens90:.4f}\n{'='*50}")

    results = {"fold": cfg.fold, "auprec": float(auprec), "auroc": float(auroc),
               "brier": float(brier), "opt_threshold": float(opt_thresh),
               "opt_f1": float(f1s[opt_idx]), "sens_at_spec90": float(sens90)}
    out_path = Path(cfg.checkpoint).parent / f"eval_fold{cfg.fold}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


# ============================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="WMA Detection Pipeline (ABCD)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("labels", help="Fuse CSV sources into binary WMA labels")
    p.add_argument("--csv1", required=True); p.add_argument("--csv2", required=True)
    p.add_argument("--out", default="data/labels_wma.csv")

    p = sub.add_parser("manifest", help="Match labels to NIfTI on disk")
    p.add_argument("--data_root", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--out", default="data/manifest.csv")

    p = sub.add_parser("train", help="Train (2-stage CE->APLoss)")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--out_dir", default="runs/wma")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--backbone", default="resnet", choices=["resnet", "swin_tiny"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--effective_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--aploss_gamma", type=float, default=0.9)
    p.add_argument("--epoch_decay", type=float, default=1e-3)
    p.add_argument("--pretrained_weights", default="")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--synthetic", action="store_true", help="Use random tensors (no NIfTI)")

    p = sub.add_parser("eval", help="Evaluate checkpoint")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_tta", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--synthetic", action="store_true", help="Use random tensors (no NIfTI)")

    args = parser.parse_args()
    if args.command == "labels":
        cmd_labels(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
