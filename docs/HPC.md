# HPC Environment — lbardou

## 1. Espace disque

| Filesystem | Capacité | Libre | Note |
|---|---|---|---|
| `/mnt/fac` (NFS) | 49 Go | ~24 Go | **Quasi-plein** — code uniquement, pas de données lourdes |
| `/mnt/scratch` (VAST) | 2.8 Po | 2.4 Po | Quasi-illimité — données + runs ici |
| `/home` (local) | 20 Go | 19 Go | **Petit** — ne pas y mettre le projet |

## 2. Paths du projet

```
# Code (cloné depuis github.com/louanbardou/wma-pipeline)
/mnt/fac/CX500007_DS1/bardou/WMA/
├── .venv/                          # venv autonome (install_env.sh)
├── activate_env.sh                 # source à chaque session
└── ...

# Données (scratch)
/mnt/scratch/user/lbardou/
├── abcd_leuko/                     # $ABCD_IMAGING — 311 Go NIfTI (T1w+T2w+dMRI)
├── wma_runs/                       # $WMA_RUNS — checkpoints, TensorBoard
├── wma_cache/                      # $WMA_CACHE — tenseurs .pt
└── wma_logs/                       # $WMA_LOGS — SLURM logs

# ABCD imaging source (read-only)
/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/
└── sub-*/ses-*/{anat,dwi}/

# RSI container
/mnt/fac/CX500002_DS1/lab-utils/    # apptainer .sif
```

## 3. Installation

```bash
cd /mnt/fac/CX500007_DS1/bardou
git clone https://github.com/louanbardou/wma-pipeline.git WMA
cd WMA
bash install_env.sh
source activate_env.sh
```

> L'ancien `leuko_env` (4.3 Go) et `leukoaraiosis-detection/` dans bardou/ sont obsolètes — peuvent être supprimés.

## 4. Outils système

Python 3.9, Git 2.47, GCC 14.2, OpenMPI 5.0, Node.js 20.20 (NVM), Claude Code, curl, wget, rsync, tmux, wandb.

## 5. Machine

- 32 cores AMD EPYC, 125 Go RAM, partagée
- Pas de GPU sur login node — soumettre via SLURM
- GPU dispo : H100 NVL (partition `gpu`)
- NE PAS utiliser `/home/remote/lbardou` (20 Go seulement) — utiliser `/mnt/fac/CX500007_DS1/bardou/`
