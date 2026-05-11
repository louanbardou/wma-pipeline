# WMA Pipeline — Full Context

Binary classification of **White Matter Abnormalities (WMA)** on pediatric brain MRI (ABCD study, ~12k children, 22 sites).
Input: T1w + T2w NIfTI (no FLAIR). Output: binary WMA 0/1. Metric: **AUPREC** (target 0.35-0.55).
Dataset: 2 717 subjects, 8 947 sessions, 739 WMA+ (8.3%), 262 unique WMA+ subjects.

---

## 1. HPC Environment (lbardou)

**Machine**: 32 cores AMD EPYC, 125 Go RAM, shared. No GPU on login — submit via SLURM. GPU: H100 NVL (partition `gpu`).
**Software**: Python 3.9, Git 2.47, GCC 14.2, OpenMPI 5.0, Node.js 20.20, Claude Code, wandb, tmux.

| Filesystem | Capacite | Note |
|---|---|---|
| `/mnt/fac` (NFS) | 49 Go | **Quasi-plein** — code only |
| `/mnt/scratch` (VAST) | 2.8 Po | Data + runs here |
| `/home` (local) | 20 Go | **Ne pas utiliser** pour le projet |

### Paths

```
# Code (git repo: github.com/louanbardou/wma-pipeline)
/mnt/fac/CX500007_DS1/bardou/wma-pipeline/
├── .venv/                          # venv (PyTorch 2.5.1/cu124, MONAI, LibAUC, nibabel)
├── activate_env.sh                 # source a chaque session (sets $ABCD_IMAGING, $WMA_DIR, $WMA_RUNS, venv, CUDA 12.5)
├── install_env.sh                  # create .venv
├── wma_pipeline.py                 # ~900L: labels, manifest, train, eval
├── wma_gradcam.py                  # ~300L: Grad-CAM heatmaps
├── rsi/                            # RSI processing (RSIproc_1_0_8.py, icosahedron.py)
├── data/
│   ├── labels_full.csv             # 8947x14 master labels
│   ├── manifest.csv                # 8947x5 pipeline-ready
│   └── available_subjects.txt      # 2717 IDs on server
├── docs/
├── jobs/
│   ├── slurm_train.sh              # 1x H100, 12 CPUs, 96G, 24h
│   ├── slurm_gradcam.sh            # 1x H100, 4 CPUs, 32G, 6h
│   └── tensorboard.sh
└── .venv/                          # gitignored

# Data (scratch — quasi-illimite)
/mnt/scratch/user/lbardou/
├── abcd_leuko/                     # $ABCD_IMAGING — 311 Go NIfTI (sub-*/ses-*/anat/*_{T1w,T2w}.nii.gz)
├── wma_runs/                       # $WMA_RUNS — checkpoints, TensorBoard
├── wma_cache/                      # $WMA_CACHE — tenseurs .pt preprocesses
└── wma_logs/                       # $WMA_LOGS — SLURM logs

# ABCD imaging source (read-only)
/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/
└── sub-*/ses-*/{anat,dwi}/

# Apptainer container (RSI)
/mnt/fac/CX500002_DS1/lab-utils/*.sif
```

### Installation

```bash
cd /mnt/fac/CX500007_DS1/bardou
git clone https://github.com/louanbardou/wma-pipeline.git WMA
cd WMA && bash install_env.sh && source activate_env.sh
```

> L'ancien `leuko_env` et `leukoaraiosis-detection/` dans bardou/ sont obsoletes.

---

## 2. Labels

**3 sources** (in `/Users/louan/Downloads/Incidental Findings/`):

| Source | Rows | Subjects | Key columns |
|---|---|---|---|
| `data_mrif_2025-10-15.csv` (Source_List/) | 37 736 | 11 855 | `id_redcap, redcap_event_name, mrif_score, mrif_fu_diff` — all subjects incl. normals |
| `abcd_combined_from-2025-10-15.csv` (Labeled_list/) | 15 451 | 6 871 | Adds `Report Text`, `Predicted Findings` (NLP) — only with findings |
| `all_labels_merged (1).csv` (/Downloads/) | 14 710 | 4 442 | `subject_id, event_name, mrif_score, finding` |
| `white_matter_IDs.csv` (Subject_ID_lists/) | 188 | 188 | Curated WMA subject list (all sessions) |

**Label=1** if ANY: `Predicted Findings` contains "WMA" OR `finding` contains "WMA" OR subject in `white_matter_IDs.csv`. Agreement: 631 both, 56 abcd_combined only, 11 all_labels only, 41 ID list only.

**mrif_score**: 0=not rated, 1=normal (7238), 2=non-significant (1300), 3=needs follow-up (379), 4=urgent (8). 26 WMA+ have score=1.

**Sessions**: baseline=2717, Y2=1959, Y3=42, Y4=1518, Y6=1806, Y8=905. BIDS: `ses-00A/02A/03A/04A/06A/08A`.
**IDs**: CSV=`NDAR_INV00HEV6HB`, BIDS=`sub-00HEV6HB` (same 8-char suffix).

### Data files (`data/`)

| File | Content |
|---|---|
| `labels_full.csv` (8947x14) | `subject_id, event_name, mrif_score, mrif_other_notes, mrif_fu_diff, session, Report Text, Predicted Findings, all_findings, wma_from_abcd_combined, wma_from_all_labels, wma_from_id_list, label, bids_id` |
| `manifest.csv` (8947x5) | `subject_id, session, t1w_path, t2w_path, label` |
| `available_subjects.txt` | 2717 subject IDs on server |

---

## 3. Model

**ResNet-3D 50** (MONAI, 23M params). Alt: Swin UNETR tiny (62M, `--backbone swin_tiny`).

**Input**: 3-channel 128^3 @ 1mm iso — [T1w, T2w, T2/T1 ratio (clamped [0,5], z-scored)].

**Head**: `AdaptiveAvgPool3d(1) -> LayerNorm(2048) -> Dropout(0.4) -> Linear(2048,256) -> GELU -> Dropout(0.2) -> Linear(256,1)`.

**Preprocessing**: LoadImage -> Orient(RAS) -> Spacing(1mm) -> CropForeground -> Normalize -> Pad/Crop(128^3). Train augmentations: flip, rotate90, affine, noise, smooth, scale intensity, cutout (all p=0.2-0.5). Val: center crop only.

---

## 4. Training

| | Stage 1 (ep 1-15) | Stage 2 (ep 16-40) |
|---|---|---|
| Loss | CE + Focal (gamma=2, alpha=0.85) | APLoss (LibAUC, gamma=0.9) |
| Optimizer | AdamW (lr=3e-4, wd=5e-4) | SOAP (lr=1e-4, epoch_decay=1e-3) |
| Sampler | Random | DualSampler (balanced) |

Progressive unfreezing (frozen ep 1-5, then 0.1x lr for backbone). EMA decay=0.9995. Grad accum=4 (effective bs=16). bf16. Early stopping patience=10 on val AUPREC. 5-fold StratifiedGroupKFold (grouped by subject).

**Metrics**: AUPREC (primary), AUROC, Brier, F1@optimal, Sens@Spec90%. Optional TTA (3 flips).

**Warning thresholds**: Stage1 AUPREC@ep15 < 0.18 -> debug. Train > 0.40 but val < 0.20 -> overfit. Site variance > 0.15 -> shortcut. Any fold > 0.65 -> leakage.

---

## 5. Grad-CAM

Target layer: `encoder.layer4[-1]`. Saves per-subject NIfTI heatmaps (WMA+ and prob > 0.5), aggregate heatmap, visualization, error analysis.

---

## 6. RSI (Restriction Spectrum Imaging)

Multi-shell diffusion MRI technique decomposing the signal into cellular compartments.

| Resource | Path |
|---|---|
| Raw dMRI | `mproc/sub-*/ses-*/dwi/*_run-01_dwi.nii.gz` |
| Apptainer container | `/mnt/fac/CX500002_DS1/lab-utils/*.sif` |
| `pyrsi` code | `github.com/rauschecker-sugrue-labs/pyrsi` (private — ask Pierre) |
| Tabulated RSI (Wynton) | `/wynton/group/abcd/{version}/tabulated/` via `abcd-utils` DataLoader |
| RSL lab data (Wynton) | `/wynton/group/rsl/ABCD_data/4.0/imaging/concatenated/` |

### RSI Metrics

| Metric | NDA Table | Description |
|---|---|---|
| **RNI** | `drsip101` | Isotropic restricted — cellularity |
| **RND** | `drsip201` | Directional restricted — fiber integrity |
| **RNT** | `drsip301` | Total restricted (RNI + RND) |
| **HNI** | `drsip401` | Hindered (extracellular) isotropic |
| **FNI** | `drsip701` | Free water fraction (CSF-like) |

Each table: ~224 vars covering WM tracts (AtlasTrack, ~44), subcortical (aseg, ~28), cortical (Desikan, ~3x68), QC.

**Variable naming**: `dmri_rsi{metric}_{atlas}_{region}_{hemisphere}`
Example: `dmri_rsi_rni_fib_at_fxcutl_lh` = RNI, fiber atlas, fornix cut left.

### RSI & WMA Relevance

| Metric | WMA relevance |
|---|---|
| RNI | Elevated in cellularity (inflammation, gliosis) |
| RND | Reduced in demyelination/axonal damage |
| FNI | Elevated in edema, free water |
| HNI | Extracellular changes in WM lesions |

### abcd-utils (Wynton)

```bash
pip install git+ssh://git@github.com:rauschecker-sugrue-labs/abcd-utils.git
```

```python
from abcd.data_loader import DataLoader
DL = DataLoader(version='5.1')
rsi_vars = DL.find_variable('dmri_rsi')
df = DL.get_variables_data(['dmri_rsi_meanmotion', 'dmri_rsi_meantrans'])
demos = DL.get_demographics()
merged = ut.abcd_merge([demos, df])
clean = ut.qc_filter(merged, fsqc=True, dmri=True)
```

### Voxel-wise RSI (needs processing)

```bash
apptainer run --bind mproc:/data /mnt/fac/CX500002_DS1/lab-utils/<container>.sif
```

> **TODO**: inspect `.sif` on CHPC, get `pyrsi` repo access from Pierre.

---

## 7. Commands Reference

### Environment

```bash
ssh lbardou@<host>
source /mnt/fac/CX500007_DS1/bardou/WMA/activate_env.sh
```

### Pipeline

```bash
# Labels (pre-built in data/ — run only if re-merging)
python wma_pipeline.py labels --csv1 <abcd_combined> --csv2 <all_labels> --out data/labels_wma.csv

# Manifest (pre-built in data/)
python wma_pipeline.py manifest --data_root $ABCD_IMAGING --labels data/labels_wma.csv --out data/manifest.csv

# Train
python wma_pipeline.py train \
    --manifest data/manifest.csv --out_dir $WMA_RUNS/<name> \
    --backbone resnet --fold 0 --epochs 40 --warmup_epochs 15 \
    --batch_size 4 --effective_batch_size 16 --lr 3e-4 \
    --cache_dir $WMA_CACHE --freeze_epochs 5 --patience 10

# Eval
python wma_pipeline.py eval --manifest data/manifest.csv --checkpoint <best_model.pt> --fold 0 --use_tta

# GradCAM
python wma_gradcam.py --checkpoint <best_model.pt> --manifest data/manifest.csv --fold 0 --out_dir heatmaps/
```

Train options: `--synthetic` (test sans NIfTI), `--no_ema`, `--pretrained_weights <path>`, `--backbone swin_tiny`.

### SLURM

```bash
sbatch jobs/slurm_train.sh                        # single fold (decomment --array=0-4 for 5)
sbatch jobs/slurm_gradcam.sh <checkpoint> <fold>
squeue -u lbardou                                 # jobs en cours
scancel <job_id>                                  # annuler
tail -f $WMA_LOGS/wma_train_<jobid>_<fold>.out    # suivre
sinfo -p gpu                                      # GPUs dispo
```

### TensorBoard

```bash
# Login node:
bash jobs/tensorboard.sh $WMA_RUNS/<name>/tb_logs
# Local:
ssh -L 6006:localhost:6006 lbardou@<host>
# Multi-fold: tensorboard --logdir_spec fold0:path0,fold1:path1,...
```

### Data exploration

```bash
grep -c ",1$" data/manifest.csv                   # count positives
wc -l data/manifest.csv                           # count lines
cut -d',' -f1 data/manifest.csv | tail -n+2 | sort -u | wc -l   # unique subjects
du -sh $ABCD_IMAGING                              # data size
ls $ABCD_IMAGING/sub-<id>/ses-<session>/anat/     # check subject files
```

### Debug

```bash
python wma_pipeline.py train --synthetic --epochs 3 --warmup_epochs 1 --batch_size 2 --fold 0
python -c "import torch; print(torch.cuda.is_available())"
python -c "from wma_pipeline import init_ml; init_ml(); print('OK')"
python -c "import torch, json; ckpt=torch.load('<path>', map_location='cpu', weights_only=False); print(ckpt['epoch'], ckpt['val_auprec'])"
```

### Git & Transfer

```bash
git add <files> && git commit -m "<msg>" && git push
rsync -avzP <local> lbardou@<host>:<remote>       # upload
rsync -avzP lbardou@<host>:<remote> <local>       # download
```
