# Commandes quotidiennes — WMA Detection Pipeline

Convention : `commande fixe` `<a remplacer>` — les `<>` sont toujours a remplacer par ta valeur.

---

## Reference rapide

```
CONNEXION & ENV
  ssh <user>@<host>                                 Connexion cluster          ex: ssh lbardou@hpc-login
  source ~/leuko.../activate_env.sh                 Activer Python + CUDA

TRANSFERT
  rsync -avzP <src> <user>@<host>:<dst>             Envoyer vers cluster       ex: rsync -avzP data/ lbardou@hpc:~/WMA/data/
  rsync -avzP <user>@<host>:<src> <dst>             Recuperer du cluster       ex: rsync -avzP lbardou@hpc:~/runs/best.pt .

PIPELINE (dans WMA/)
  python wma_pipeline.py labels   --csv1 <f> --csv2 <f> --out <f>              Fusionner CSV -> labels WMA
  python wma_pipeline.py manifest --data_root <dir> --labels <f> --out <f>     Matcher labels aux NIfTI
  python wma_pipeline.py train    --manifest <f> --out_dir <dir> --fold <0-4>  Entrainement 2-stage
  python wma_pipeline.py eval     --checkpoint <f> --fold <0-4> [--use_tta]    Evaluer un checkpoint
  python wma_gradcam.py           --checkpoint <f> --fold <0-4> --out_dir <d>  Heatmaps GradCAM

SLURM
  sbatch jobs/slurm_train.sh                        Soumettre training         (decommenter --array=0-4 pour 5 folds)
  sbatch jobs/slurm_gradcam.sh <checkpoint> <fold>  Soumettre GradCAM          ex: sbatch jobs/slurm_gradcam.sh runs/best.pt 0
  squeue -u <user>                                  Jobs en cours              ex: squeue -u lbardou
  scancel <job_id>                                  Annuler un job             ex: scancel 149872
  tail -f <log>                                     Suivre en temps reel       ex: tail -f /mnt/.../wma_train_149872_0.out

TENSORBOARD (2 terminaux)
  bash jobs/tensorboard.sh <dir>                    Login node : lancer TB     ex: bash jobs/tensorboard.sh /mnt/.../leuko_runs
  ssh -L 6006:localhost:6006 <user>@<host>          Local : tunnel SSH         puis http://localhost:6006

GIT
  git add <files> && git commit -m "<msg>"          Commit                     ex: git add WMA/ && git commit -m "fix bug"
  git push / git pull                               Pousser / recuperer

DONNEES
  grep -c ",1$" <csv>                               Compter positifs           ex: grep -c ",1$" data/manifest.csv
  wc -l <csv>                                       Compter lignes             ex: wc -l data/manifest.csv
  du -sh <dir>                                      Taille dossier             ex: du -sh /mnt/.../leuko_cache_wma
  sinfo -p gpu                                      GPUs disponibles

DEBUG
  python wma_pipeline.py train --synthetic --epochs 3    Test sans NIfTI
  python -c "import torch; print(torch.cuda.is_available())"   Check GPU
```

---

## Details par section

---

## 1. Connexion & environnement

```bash
# SSH vers le cluster
ssh <user>@<login-node>
#     <user>       : ton username HPC (lbardou)
#     <login-node> : adresse du login node

# Activer l'environnement Python + CUDA
source ~/WMA/activate_env.sh
```

---

## 2. Transfert de fichiers

```bash
# Envoyer des fichiers locaux vers le cluster
rsync -avzP <fichier_local> <user>@<login-node>:<chemin_distant>
#     <fichier_local>  : chemin sur ta machine (ex: data/labels.csv)
#     <chemin_distant>  : destination sur le cluster (ex: ~/leukoaraiosis-detection/WMA/data/)

# Recuperer des fichiers du cluster vers local
rsync -avzP <user>@<login-node>:<chemin_distant> <dossier_local>
#     <chemin_distant>  : fichier sur le cluster (ex: ~/leuko_runs/wma_fold0/best_model.pt)
#     <dossier_local>   : destination locale (ex: ./results/)

# Synchroniser un dossier entier
rsync -avzP --exclude '*.pt' --exclude '__pycache__' \
    <dossier_local>/ <user>@<login-node>:<dossier_distant>/
#     --exclude : patterns a ignorer (checkpoints lourds, cache python)
```

---

## 3. Pipeline WMA

### 3.1 Preparer les labels

```bash
python wma_pipeline.py labels \
    --csv1 <chemin_csv1> \
    --csv2 <chemin_csv2> \
    --out <chemin_sortie>
#     <chemin_csv1>    : CSV avec colonne "Predicted Findings" (abcd_combined_from-2025-10-15.csv)
#     <chemin_csv2>    : CSV avec colonne "finding" (all_labels_merged.csv)
#     <chemin_sortie>  : ou sauver le CSV fusionne (ex: data/labels_wma.csv)
```

### 3.2 Construire le manifest

```bash
python wma_pipeline.py manifest \
    --data_root <dossier_nifti> \
    --labels <csv_labels> \
    --out <chemin_sortie>
#     <dossier_nifti> : racine des NIfTI (ex: /mnt/scratch/user/lbardou/abcd_leuko)
#                       doit contenir sub-*/ses-*/anat/*T1w.nii.gz et *T2w.nii.gz
#     <csv_labels>    : labels generes a l'etape precedente (data/labels_wma.csv)
#     <chemin_sortie> : manifest CSV de sortie (ex: data/manifest.csv)
```

### 3.3 Lancer le training

```bash
python wma_pipeline.py train \
    --manifest <csv_manifest> \
    --out_dir <dossier_sortie> \
    --backbone <architecture> \
    --epochs <nombre> \
    --fold <numero> \
    --cache_dir <dossier_cache>
#     <csv_manifest>    : manifest genere a l'etape precedente (data/manifest.csv)
#     <dossier_sortie>  : ou sauver modele + logs (ex: runs/wma_fold0)
#     <architecture>    : "resnet" (14.5M, recommande) ou "swin_tiny" (15.8M)
#     <nombre>          : nombre d'epochs total (defaut: 40)
#     <numero>          : index du fold 0-4 pour la cross-validation
#     <dossier_cache>   : dossier scratch pour cacher les tenseurs preprocesses
#                         (ex: /mnt/scratch/user/lbardou/leuko_cache_wma)
```

Options supplementaires :
```
    --warmup_epochs <n>            : epochs de CE warmup avant APLoss (defaut: 15)
    --batch_size <n>               : batch physique GPU (defaut: 4)
    --effective_batch_size <n>     : batch effectif via gradient accumulation (defaut: 16)
    --lr <float>                   : learning rate initial (defaut: 3e-4)
    --freeze_epochs <n>            : epochs avec backbone gele (defaut: 5)
    --pretrained_weights <chemin>  : checkpoint pre-entraine (.pth)
    --no_ema                       : desactiver l'EMA des poids
    --patience <n>                 : early stopping patience (defaut: 10)
    --synthetic                    : tenseurs aleatoires pour test sans NIfTI
```

### 3.4 Evaluer un checkpoint

```bash
python wma_pipeline.py eval \
    --manifest <csv_manifest> \
    --checkpoint <chemin_checkpoint> \
    --fold <numero>
#     <chemin_checkpoint> : chemin vers best_model.pt (ex: runs/wma_fold0/best_model.pt)
#     <numero>            : meme fold que celui utilise pour l'entrainement

    --use_tta      : activer le test-time augmentation (4 passes, +1-3% AUPREC)
    --synthetic    : mode test sans NIfTI
```

### 3.5 GradCAM

```bash
python wma_gradcam.py \
    --checkpoint <chemin_checkpoint> \
    --manifest <csv_manifest> \
    --fold <numero> \
    --out_dir <dossier_sortie>
#     <dossier_sortie> : ou sauver les heatmaps NIfTI (ex: heatmaps/fold0/)

# Sortie :
#   - sub-XXX_ses-YYY_gradcam.nii.gz   (heatmap par sujet)
#   - aggregate_gradcam_positives.nii.gz (moyenne sur les WMA+)
#   - aggregate_gradcam.png              (visualisation 3 plans)
#   - error_analysis.txt                 (top faux positifs / faux negatifs)
```

---

## 4. SLURM — gestion des jobs

### 4.1 Soumettre un job

```bash
# Training single fold
sbatch jobs/slurm_train.sh

# Training 5 folds en parallele (decommenter --array=0-4 dans le script)
sbatch jobs/slurm_train.sh

# GradCAM
sbatch jobs/slurm_gradcam.sh <chemin_checkpoint> <fold>
#     <chemin_checkpoint> : best_model.pt du run
#     <fold>              : numero du fold (0-4)
```

### 4.2 Surveiller les jobs

```bash
# Voir tous tes jobs en cours
squeue -u <user>
#     <user> : ton username (lbardou)

# Details d'un job specifique
scontrol show job <job_id>
#     <job_id> : numero du job (ex: 149872)

# Suivre la sortie en temps reel
tail -f <fichier_log>
#     <fichier_log> : chemin du .out SLURM
#                     (ex: /mnt/scratch/user/lbardou/leuko_logs/wma_train_149872_0.out)

# Annuler un job
scancel <job_id>

# Annuler tous tes jobs
scancel -u <user>
```

### 4.3 Ressources et quotas

```bash
# Voir les GPUs disponibles
sinfo -p gpu --Format="NodeHost,Gres,StateLong,CPUsState,Memory"

# Voir ta consommation
sacct -u <user> --starttime=<date> --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State
#     <date> : date de debut (ex: 2026-05-01)

# Verifier l'espace disque
df -h /mnt/scratch/user/<user>
du -sh /mnt/scratch/user/<user>/*
```

---

## 5. TensorBoard — suivi en temps reel

```bash
# Sur le login node (pendant que le job tourne) :
bash jobs/tensorboard.sh <dossier_logs>
#     <dossier_logs> : racine des runs (ex: /mnt/scratch/user/lbardou/leuko_runs)

# Sur ta machine locale (dans un autre terminal) :
ssh -L 6006:localhost:6006 <user>@<login-node>
# Puis ouvrir http://localhost:6006

# Si le port 6006 est occupe :
ssh -L <port_local>:localhost:<port_distant> <user>@<login-node>
#     <port_local>   : port libre sur ta machine (ex: 6007)
#     <port_distant>  : port passe a tensorboard (ex: 6007)
```

---

## 6. Git

```bash
# Voir l'etat des modifications
git status

# Ajouter des fichiers specifiques
git add <fichiers>
#     <fichiers> : chemins des fichiers modifies (ex: WMA/wma_pipeline.py WMA/progress.md)

# Commiter
git commit -m "<message>"
#     <message> : description courte du changement (ex: "fix label=0 bug in manifest builder")

# Pousser
git push

# Recuperer les derniers changements (si tu travailles depuis plusieurs machines)
git pull

# Voir l'historique recent
git log --oneline -<n>
#     <n> : nombre de commits a afficher (ex: 10)

# Voir ce qui a change dans un fichier
git diff <fichier>
```

---

## 7. Exploration des donnees

```bash
# Compter les lignes d'un CSV
wc -l <fichier_csv>

# Voir les premieres lignes
head -n <n> <fichier_csv>

# Compter les positifs dans un manifest
grep -c ",1$" <fichier_csv>

# Nombre de sujets uniques
cut -d',' -f1 <fichier_csv> | tail -n+2 | sort -u | wc -l

# Verifier qu'un sujet a bien T1+T2 sur disque
ls <dossier_nifti>/sub-<id>/ses-<session>/anat/
#     <id>      : ID du sujet (ex: 0A4ZDYNL)
#     <session> : session (ex: 00A)

# Taille totale des donnees
du -sh <dossier_nifti>
```

---

## 8. Debug rapide

```bash
# Tester la pipeline sans donnees reelles (synthetic)
python wma_pipeline.py train \
    --manifest data/manifest_synthetic.csv \
    --out_dir runs/test \
    --epochs 3 --warmup_epochs 1 --batch_size 2 \
    --fold 0 --synthetic

# Verifier que les imports fonctionnent
python -c "from wma_pipeline import init_ml; init_ml(); print('OK')"

# Verifier les GPU visibles (sur un noeud de calcul)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Verifier la version de libauc
python -c "import libauc; print(libauc.__version__)"

# Inspecter un checkpoint
python -c "
import torch, json
ckpt = torch.load('<chemin_checkpoint>', map_location='cpu', weights_only=False)
print('Epoch:', ckpt['epoch'])
print('Val AUPREC:', ckpt['val_auprec'])
print('Config:', json.dumps(ckpt['config'], indent=2))
"
#     <chemin_checkpoint> : best_model.pt a inspecter
```

---

## 9. Visualisation des resultats

```bash
# Ouvrir un heatmap NIfTI (si fsleyes est installe)
fsleyes <fichier_nifti>

# Voir les metriques d'un eval
cat <dossier_run>/eval_fold<n>.json
#     <n> : numero du fold

# Voir l'historique d'entrainement
cat <dossier_run>/training_history.csv

# Comparer les metriques de plusieurs folds
for f in 0 1 2 3 4; do
    echo "Fold $f: $(cat <dossier_run_base>_fold${f}/eval_fold${f}.json | python -c 'import sys,json; d=json.load(sys.stdin); print(f"AUPREC={d[\"auprec\"]:.4f} AUROC={d[\"auroc\"]:.4f}")')"
done
```
