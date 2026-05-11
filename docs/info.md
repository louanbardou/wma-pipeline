# WMA Pipeline v2 — Reference

Binary classification of **White Matter Abnormalities (WMA)** on pediatric brain MRI (ABCD study, ~12k children, 22 sites).
Input: T1w + T2w NIfTI (no FLAIR). Output: binary WMA 0/1. Metric: **AUPREC** (target 0.35-0.55).
Dataset: 2 717 subjects, 8 947 sessions, 739 WMA+ (8.3%), 262 unique WMA+ subjects.

---

## 1. Labels

**3 sources** (in `/Users/louan/Downloads/Incidental Findings/`):

| Source | Rows | Subjects | Key columns |
|---|---|---|---|
| `data_mrif_2025-10-15.csv` (Source_List/) | 37 736 | 11 855 | `id_redcap, redcap_event_name, mrif_score, mrif_fu_diff` — **all** subjects incl. normals |
| `abcd_combined_from-2025-10-15.csv` (Labeled_list/) | 15 451 | 6 871 | Adds `Report Text`, `Predicted Findings` (NLP) — only subjects with findings |
| `all_labels_merged (1).csv` (/Downloads/) | 14 710 | 4 442 | `subject_id, event_name, mrif_score, finding` |
| `white_matter_IDs.csv` (Subject_ID_lists/) | 188 | 188 | Curated WMA subject list (applies to all sessions) |

**Label=1 (WMA+)** if ANY: `Predicted Findings` contains "WMA" OR `finding` contains "WMA" OR subject in `white_matter_IDs.csv`. Agreement: 631 both, 56 abcd_combined only, 11 all_labels only, 41 ID list only.

**mrif_score**: 0=not rated, 1=normal (7238), 2=non-significant finding (1300), 3=needs follow-up (379), 4=urgent (8). 26 WMA+ have score=1 (mild, flagged by NLP).

**Sessions**: baseline=2717, Y2=1959, Y3=42, Y4=1518, Y6=1806, Y8=905. BIDS: `ses-00A/02A/03A/04A/06A/08A`.

**IDs**: CSV=`NDAR_INV00HEV6HB`, BIDS=`sub-00HEV6HB` (same 8-char suffix).

---

## 2. Data files (`WMA/data/`)

| File | Content |
|---|---|
| `labels_full.csv` (8947×14) | Master: `subject_id, event_name, mrif_score, mrif_other_notes, mrif_fu_diff, session, Report Text, Predicted Findings, all_findings, wma_from_abcd_combined, wma_from_all_labels, wma_from_id_list, label, bids_id` |
| `manifest.csv` (8947×5) | Pipeline-ready: `subject_id, session, t1w_path, t2w_path, label` |
| `available_subjects.txt` | 2717 subject IDs on server |

---

## 3. HPC paths

| Path | Content |
|---|---|
| `/mnt/fac/CX500007_DS1/bardou/wma-pipeline/` | Pipeline code + data + jobs (git repo) |
| `/mnt/scratch/user/lbardou/abcd_leuko/` | NIfTI data (`$ABCD_IMAGING`): `sub-*/ses-*/anat/*_{T1w,T2w}.nii.gz` |
| `/mnt/scratch/user/lbardou/wma_runs/` | Checkpoints, TensorBoard logs (`$WMA_RUNS`) |
| `/mnt/scratch/user/lbardou/wma_cache/` | Preprocessed `.pt` cache (`$WMA_CACHE`) |
| `/mnt/scratch/user/lbardou/wma_logs/` | SLURM logs (`$WMA_LOGS`) |
| `wma-pipeline/.venv/` | Python venv (PyTorch 2.5.1/cu124, MONAI, LibAUC, nibabel). Install: `bash install_env.sh` |
| `/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/` | Original ABCD imaging (read-only, includes `anat/` + `dwi/`) |

Activate: `source activate_env.sh` (sets `$ABCD_IMAGING`, `$WMA_DIR`, `$WMA_RUNS`, venv, CUDA 12.5).

**TensorBoard**: logs are in `$WMA_RUNS/<run_name>/tb_logs/`. Launch with:
```bash
tensorboard --logdir /mnt/scratch/user/lbardou/wma_runs/<run_name>/tb_logs --bind_all --port 6006
```
Then tunnel: `ssh -L 6006:localhost:6006 lbardou@<host>`. For multiple folds use `--logdir_spec fold0:path0,fold1:path1,...`.

---

## 4. Pipeline

**Files**: `wma_pipeline.py` (~900L, subcommands: `labels`, `manifest`, `train`, `eval`), `wma_gradcam.py` (~300L).

```bash
python wma_pipeline.py labels --csv1 <abcd_combined> --csv2 <all_labels> --out data/labels_wma.csv
python wma_pipeline.py manifest --data_root $ABCD_IMAGING --labels data/labels_wma.csv --out data/manifest.csv
python wma_pipeline.py train --manifest data/manifest.csv --out_dir runs/wma --backbone resnet --fold 0
python wma_pipeline.py eval --manifest data/manifest.csv --checkpoint runs/wma/best_model.pt --fold 0 --use_tta
```

Note: `labels_full.csv` and `manifest.csv` are **pre-built** in `WMA/data/` (3-source merge done externally).

---

## 5. Model

**ResNet-3D 50** (MONAI, 23M params). Alt: Swin UNETR tiny (62M, `--backbone swin_tiny`).

**Input**: 3-channel 128³ @ 1mm iso — [T1w, T2w, T2/T1 ratio (clamped [0,5], z-scored)].

**Head**: `AdaptiveAvgPool3d(1) → LayerNorm(2048) → Dropout(0.4) → Linear(2048,256) → GELU → Dropout(0.2) → Linear(256,1)`.

**Preprocessing**: LoadImage → Orient(RAS) → Spacing(1mm) → CropForeground → Normalize → Pad/Crop(128³). Train augmentations: flip, rotate90, affine, noise, smooth, scale intensity, cutout (all p=0.2-0.5). Val: center crop only.

---

## 6. Training

| | Stage 1 (ep 1-15) | Stage 2 (ep 16-40) |
|---|---|---|
| Loss | CE + Focal (gamma=2, alpha=0.85) | APLoss (LibAUC, gamma=0.9) |
| Optimizer | AdamW (lr=3e-4, wd=5e-4) | SOAP (lr=1e-4, epoch_decay=1e-3) |
| Sampler | Random | DualSampler (balanced) |

Progressive unfreezing (frozen ep 1-5, then 0.1x lr for backbone). EMA decay=0.9995. Grad accum=4 (effective bs=16). bf16. Early stopping patience=10 on val AUPREC. 5-fold StratifiedGroupKFold (grouped by subject).

**Metrics**: AUPREC (primary), AUROC, Brier, F1@optimal, Sens@Spec90%. Optional TTA (3 flips).

**Warning thresholds**: Stage1 AUPREC@ep15 < 0.18 → debug. Train > 0.40 but val < 0.20 → overfit. Site variance > 0.15 → shortcut. Any fold > 0.65 → leakage.

---

## 7. Grad-CAM

Target layer: `encoder.layer4[-1]`. Saves per-subject NIfTI heatmaps (WMA+ and prob > 0.5), aggregate heatmap, visualization, error analysis.

```bash
python wma_gradcam.py --checkpoint runs/wma/best_model.pt --manifest data/manifest.csv --fold 0 --out_dir heatmaps/
```

---

## 8. SLURM

| Job | GPU | CPUs | RAM | Time | Script |
|---|---|---|---|---|---|
| Training | 1x H100 NVL | 12 | 96G | 24h | `jobs/slurm_train.sh` |
| GradCAM | 1x H100 NVL | 4 | 32G | 6h | `jobs/slurm_gradcam.sh` |

5-fold: uncomment `--array=0-4`. TensorBoard: `bash jobs/tensorboard.sh`, then `ssh -L 6006:localhost:6006`.

---

## 9. RSI (Restriction Spectrum Imaging)

| Resource | Path |
|---|---|
| Raw dMRI | `mproc/sub-*/ses-*/dwi/*_run-01_dwi.nii.gz` |
| Apptainer container | `/mnt/fac/CX500002_DS1/lab-utils/*.sif` |
| `pyrsi` code | `github.com/rauschecker-sugrue-labs/pyrsi` (private — ask Pierre) |
| Tabulated RSI | Wynton `/wynton/group/abcd/{version}/tabulated/` via `abcd-utils` DataLoader |

**Metrics**: RNI (cellularity, `drsip101`), RND (fiber integrity, `drsip201`), RNT (total, `drsip301`), HNI (extracellular, `drsip401`), FNI (free water, `drsip701`). ~224 vars each (tracts + aseg + Desikan).

**Tabulated** (ready): `DataLoader(version='5.1').find_variable('dmri_rsi')`.
**Voxel-wise** (needs processing): `apptainer run --bind mproc:/data /mnt/fac/CX500002_DS1/lab-utils/<container>.sif`.

> **TODO**: inspect `.sif` on CHPC, get `pyrsi` repo access from Pierre.

---

## 10. Directory tree

```
wma-pipeline/                # Self-contained — clone this alone
├── activate_env.sh          # Environment activation
├── install_env.sh           # Create .venv with all dependencies
├── requirements.txt         # Pip dependencies
├── .gitignore
├── wma_pipeline.py          # labels, manifest, train, eval
├── wma_gradcam.py           # Grad-CAM heatmaps
├── rsi/                     # RSI processing (from ABCD-STUDY/RSI, MIT)
│   ├── RSIproc_1_0_8.py     # dwi → RSI maps (RNI, RND, FNI, etc.)
│   └── icosahedron.py       # dependency
├── data/{labels_full,manifest}.csv, available_subjects.txt
├── docs/{info,abcd,rsi_processing,progress,HPC}.md
├── jobs/{slurm_train,slurm_gradcam,tensorboard}.sh
└── .venv/                   # created by install_env.sh (gitignored)
```
