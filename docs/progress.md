# WMA Pipeline v2 — Progress

## Objectif
Detection binaire de White Matter Abnormalities (WMA) chez les enfants (ABCD dataset).
Cible realiste : AUPREC 0.35-0.55.

## Fichiers

| Fichier | Role | Status |
|---|---|---|
| `wma_pipeline.py` | Pipeline complete (labels, manifest, model, train, eval) | DONE |
| `wma_gradcam.py` | Visualisation GradCAM post-training | DONE |
| `jobs/slurm_train.sh` | Job SLURM training (single + array multi-fold) | DONE |
| `jobs/slurm_gradcam.sh` | Job SLURM GradCAM | DONE |
| `jobs/tensorboard.sh` | Lancer TensorBoard sur login node | DONE |
| `data/labels_full.csv` | Labels 8947x14 (3-source merge) | DONE |
| `data/manifest.csv` | Manifest 8947x5 avec paths NIfTI | DONE |
| `data/available_subjects.txt` | 2717 IDs disponibles sur serveur | DONE |
| `docs/context.md` | Doc unique : env, labels, modele, training, RSI, paths | DONE |
| `docs/commands.md` | Reference commandes detaillees | DONE |
| `progress.md` | Ce fichier | Actif |

## Architecture du pipeline

```
wma_pipeline.py
├── labels   — Fusionne 2 CSV → labels binaires WMA (0/1)
├── manifest — Match labels aux NIfTI sur disque (T1+T2)
├── train    — 2-stage training (CE+Focal → APLoss+SOAP)
└── eval     — Evaluation avec TTA, seuil optimal, metriques

wma_gradcam.py
└── Genere heatmaps NIfTI + visualisations par sujet
```

## Changements cles vs pipeline v1

| # | Changement | Impact |
|---|---|---|
| 1 | ResNet-3D ~23M params (au lieu de Swin UNETR 62M) | CRITIQUE |
| 2 | 3eme canal T2/T1 ratio | ELEVE |
| 3 | Training 2-stage: CE+Focal warmup (15 ep) → APLoss+SOAP (25 ep) | ELEVE |
| 4 | TensorBoard offline (remplace W&B) | UX |
| 5 | bf16 mixed precision | VITESSE |
| 6 | Gradient accumulation (eff. batch 16) | MODERE |
| 7 | EMA des poids (decay 0.9995) | MODERE |
| 8 | Fix scheduler cosine (step par epoch) | MODERE |
| 9 | Early stopping (patience 10) | SECURITE |
| 10 | CutOut 3D augmentation | MODERE |

## Comment lancer

### 1. Preparer les labels (une seule fois)
```bash
python wma_pipeline.py labels \
    --csv1 ../abcd_combined_from-2025-10-15.csv \
    --csv2 "../all_labels_merged (1).csv" \
    --out data/labels_wma.csv
```

### 2. Construire le manifest (une seule fois, sur le HPC)
```bash
python wma_pipeline.py manifest \
    --data_root $ABCD_IMAGING \
    --labels data/labels_wma.csv \
    --out data/manifest.csv
```

### 3. Training (via SLURM)
```bash
# Single fold
sbatch jobs/slurm_train.sh

# Multi-fold (5 folds en parallele)
# Decommenter #SBATCH --array=0-4 dans slurm_train.sh
sbatch jobs/slurm_train.sh
```

### 4. Monitorer (TensorBoard)
```bash
# Sur le login node :
bash jobs/tensorboard.sh /mnt/scratch/user/lbardou/leuko_runs/

# Sur ta machine locale :
ssh -L 6006:localhost:6006 lbardou@hpc-login
# Ouvrir http://localhost:6006
```

### 5. GradCAM
```bash
sbatch jobs/slurm_gradcam.sh /path/to/best_model.pt 0
```

## Early warning signals

| Signal | Seuil | Action |
|---|---|---|
| Stage 1 (ep 15) val AUPREC | < 0.18 | Stop, debug backbone/data |
| Train AUPREC vs val | train > 0.40, val < 0.20 | Overfit: reduce model size |
| Per-site AUPREC variance | > 0.15 | Site shortcut: ajouter GRL |
| Val AUPREC single fold | > 0.65 | Suspecter leak, auditer folds |

## Log

### 2026-05-10
- [x] Creation structure WMA/
- [x] Ecriture wma_pipeline.py (sections 0-9, ~900 lignes)
- [x] Ecriture wma_gradcam.py (~300 lignes)
- [x] Ecriture jobs SLURM (train, gradcam, tensorboard)
- [x] Ecriture progress.md
- [x] Fix bug label=0 falsy dans manifest builder (`or` → `if None`)
- [x] Fix `encoder.relu` → `encoder.act` (MONAI ResNet)
- [x] Ajout mode `--synthetic` pour test sans NIfTI
- [x] Tests dry-run valides :
  - `labels` : 1040 WMA+ / 14411 WMA- / 438 sujets uniques
  - `manifest` : labels 0 et 1 correctement matches
  - `train` resnet : 4 epochs OK, checkpoint + TB logs + history CSV
  - `train` swin_tiny : 3 epochs OK
  - `train` avec EMA : OK
  - `eval` : metriques OK (AUPREC, AUROC, Brier, seuil optimal, Sens@Spec90)
  - `eval` avec TTA : OK
  - `wma_gradcam.py` : heatmaps NIfTI + aggregate + plots + error analysis OK
  - Seul non-teste : libauc (APLoss/SOAP/DualSampler) — pas installe localement, sera teste sur HPC

### 2026-05-11
- [x] Fusion docs (abcd.md + HPC.md + info.md + commands.md) → `docs/context.md` (document unique compact)
- [x] Suppression abcd.md, HPC.md, info.md (redondants)
- [x] Conservation commands.md (reference commandes detaillees)
- [x] Mise a jour progress.md (table fichiers, log)
