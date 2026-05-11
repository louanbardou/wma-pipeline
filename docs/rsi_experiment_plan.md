# Experiment Plan: WMA Detection via RSI Signature

**Objectif papier**: montrer que les métriques RSI (RNI, RND, FNI) présentent une signature microstructurelle distincte dans les WMA pédiatriques, potentiellement détectable avant ou au-delà de ce que T1/T2 montre seul.

**Cohorte pilote**: 6 WMA+ (mrif_score=3) vs 6 sains (mrif_score=1), tous baseline ses-00A, tous avec dMRI dans mproc.

---

## Etape 1 — Compute RSI maps (serveur, ~3h)

Pour les 12 sujets, produire les cartes RSI voxel-wise à partir des dMRI brutes.

```bash
source /mnt/fac/CX500007_DS1/bardou/WMA/activate_env.sh
cd /mnt/fac/CX500007_DS1/bardou/WMA

MPROC=/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc
OUT=/mnt/scratch/user/$USER/rsi_output

for SUBJ in 9PJ7VRDA 61PF7E1L 80VAXPN1 93JXKKF3 99TVX9G8 701F04JM \
            2HLV1V0P 2HLV10CC 2HLZM8RB 2J3D85NJ 2JB8MUAJ 2K3JH38W; do
    DWI_DIR=$MPROC/sub-$SUBJ/ses-00A/dwi
    DWI=$(ls $DWI_DIR/*_dwi.nii.gz 2>/dev/null | head -1)
    [ -z "$DWI" ] && echo "SKIP $SUBJ: no dwi" && continue

    BVAL=$(ls $DWI_DIR/*.bval 2>/dev/null | head -1)
    BVEC=$(ls $DWI_DIR/*.bvec 2>/dev/null | head -1)

    mkdir -p $OUT/sub-$SUBJ && cd $OUT/sub-$SUBJ
    fslroi $DWI b0 0 1
    bet b0 b0_brain -m -f 0.3
    python $WMA_DIR/rsi/RSIproc_1_0_8.py $DWI b0_brain_mask.nii.gz $BVAL $BVEC
    echo "Done: $SUBJ"
done
```

**Output par sujet**: `n0s1.nii.gz` (RNI), `nds1.nii.gz` (RND), `n0s3.nii.gz` (FNI), + autres.

---

## Etape 2 — Segmentation manuelle des WMA (ITK-SNAP, ~3h)

Pour les 6 WMA+ uniquement. Segmenter sur T1+T2 (pas sur RSI — le RSI est ce qu'on évalue).

### Protocole

1. **Espace**: segmenter dans l'espace T1 natif
2. **Critères d'inclusion**: hyperintense T2, hypo/iso T1, visible sur 2 plans, ≥3 voxels, dans la substance blanche
3. **Critères d'exclusion**: espaces périvasculaires, myélinisation tardive péri-trigonale, artefacts
4. **Outil**: ITK-SNAP, T1 en image principale, T2 en overlay

```bash
# Pour chaque WMA+
itksnap -g sub-XX_T1w.nii.gz -o sub-XX_T2w.nii.gz
# Paintbrush (B), label 1, rayon 1mm
# Sauvegarder → sub-XX_lesion_mask.nii.gz
```

### Validation
- Re-segmenter 2 patients à J+7 en aveugle → Dice intra-rater ≥ 0.7
- Validation par neuroradiologue UCSF

---

## Etape 3 — Registration masques → espace DWI (~30 min)

Les masques sont en espace T1. Les RSI maps sont en espace DWI. Il faut aligner.

```bash
# Pour chaque sujet WMA+
flirt -in sub-XX_T1w.nii.gz -ref sub-XX_b0_brain.nii.gz \
      -omat T1_to_dwi.mat -dof 6 -cost normmi

flirt -in sub-XX_lesion_mask.nii.gz -ref sub-XX_b0_brain.nii.gz \
      -applyxfm -init T1_to_dwi.mat \
      -interp nearestneighbour \
      -out sub-XX_lesion_in_dwi.nii.gz
```

---

## Etape 4 — Extraction des métriques RSI (Python, ~1h)

Pour chaque WMA+, extraire les valeurs RSI dans la lésion vs dans la NAWM (Normal-Appearing White Matter) du même patient.

```python
import nibabel as nib
import numpy as np
import pandas as pd

subjects_wma = ['9PJ7VRDA', '61PF7E1L', '80VAXPN1', '93JXKKF3', '99TVX9G8', '701F04JM']
subjects_ctrl = ['2HLV1V0P', '2HLV10CC', '2HLZM8RB', '2J3D85NJ', '2JB8MUAJ', '2K3JH38W']
metrics = {'RNI': 'n0s1.nii.gz', 'RND': 'nds1.nii.gz', 'FNI': 'n0s3.nii.gz'}

rows = []
for subj in subjects_wma:
    lesion = nib.load(f'{subj}/lesion_in_dwi.nii.gz').get_fdata() > 0
    # NAWM = WM mask minus lesion (FSL FAST sur T1 transformé en DWI space)
    wm = nib.load(f'{subj}/wm_mask_in_dwi.nii.gz').get_fdata() > 0
    nawm = wm & ~lesion

    for name, fname in metrics.items():
        data = nib.load(f'{subj}/{fname}').get_fdata()
        rows.append({'subject': subj, 'group': 'WMA+', 'region': 'lesion',
                     'metric': name, 'mean': data[lesion].mean(), 'std': data[lesion].std()})
        rows.append({'subject': subj, 'group': 'WMA+', 'region': 'NAWM',
                     'metric': name, 'mean': data[nawm].mean(), 'std': data[nawm].std()})

for subj in subjects_ctrl:
    wm = nib.load(f'{subj}/wm_mask_in_dwi.nii.gz').get_fdata() > 0
    for name, fname in metrics.items():
        data = nib.load(f'{subj}/{fname}').get_fdata()
        rows.append({'subject': subj, 'group': 'Ctrl', 'region': 'WM',
                     'metric': name, 'mean': data[wm].mean(), 'std': data[wm].std()})

df = pd.DataFrame(rows)
df.to_csv('rsi_lesion_analysis.csv', index=False)
```

---

## Etape 5 — Analyses statistiques (Python, ~2h)

### 5.1 Comparaison intra-patient : lésion vs NAWM

La question centrale. Pour chaque WMA+, comparer les distributions RSI dans la lésion vs la NAWM du même patient. Paired test.

```
Pour chaque métrique (RNI, RND, FNI):
  - Wilcoxon signed-rank test (paired, n=6)
  - Effect size (Cohen's d ou rank-biserial)
  - Direction attendue: lésion a RNI↑, RND↓, FNI↑ vs NAWM
```

### 5.2 Comparaison inter-groupe : WM des WMA+ vs WM des contrôles

Est-ce que même la NAWM des patients WMA+ est différente de la WM des sains ? (lésion invisible mais microstructure altérée — c'est l'argument du papier pour RSI > T1/T2).

```
Mann-Whitney U test: NAWM(WMA+) vs WM(Ctrl), pour chaque métrique
```

### 5.3 Profil RSI des lésions (figure clé du papier)

Bar plot ou violin plot montrant pour chaque métrique :
- WM contrôles | NAWM WMA+ | Lésion WMA+

→ Si NAWM WMA+ ≠ WM contrôles en RSI alors que T1/T2 ne montrent rien dans cette zone, c'est l'argument que RSI détecte du pathologique invisible en conventionnel.

---

## Etape 6 — Figures (~2h)

### Figure 1: Cas illustratif
Un patient WMA+ avec side-by-side: T1w | T2w | RNI map | RND map | FNI map
Avec le masque lésionnel en contour rouge.
Montrer que la lésion est visible en T2 ET en RNI/FNI, et que RND montre une perte de signal.

### Figure 2: Boxplots RSI
3 panneaux (RNI, RND, FNI) × 3 groupes (WM ctrl, NAWM WMA+, Lésion WMA+).
Avec les p-values des tests.

### Figure 3 (si résultat positif): Heatmap de sensibilité
ROC curves pour la classification lésion vs NAWM basée sur chaque métrique RSI seule.
Comparer à un seuil T2 intensity-based.

---

## Timeline

| Etape | Quoi | Durée | Pré-requis |
|---|---|---|---|
| 1 | Compute RSI maps (12 sujets) | 3h (serveur) | Vérifier bval/bvec existent |
| 2 | Segmentation manuelle (6 WMA+) | 3h (ITK-SNAP) | Installer ITK-SNAP |
| 3 | Registration masques → DWI | 30 min | FSL installé |
| 4 | Extraction métriques | 1h | Etapes 1-3 done |
| 5 | Stats | 2h | Etape 4 done |
| 6 | Figures | 2h | Etape 5 done |
| **Total** | | **~12h** | étalable sur 1 semaine |

---

## Go / No-Go après l'étape 5

- **Go (papier viable)**: différence significative lésion vs NAWM en RSI (p < 0.05) ET/OU NAWM WMA+ ≠ WM ctrl en RSI
- **Super-go**: NAWM diffère en RSI alors qu'elle est normale en T1/T2 → RSI détecte du "pré-lésionnel"
- **No-go**: aucune différence en RSI → les WMA pédiatriques n'ont pas de signature microstructurelle RSI, pivot vers autre chose

## Points de vigilance

1. **Segmenter sur T1/T2, analyser en RSI** — ne jamais regarder les cartes RSI pendant la segmentation, sinon biais circulaire
2. **La registration T1→DWI doit être vérifiée visuellement** — une mauvaise registration rend l'analyse entière invalide
3. **N=6 est un pilote** — pas de correction pour tests multiples, les résultats orientent mais ne prouvent pas. Le papier final nécessitera plus de sujets
4. **Le "mieux que T1/T2" n'est pas l'argument principal** — l'argument est que RSI donne une info COMPLÉMENTAIRE (microstructure vs morphologie). Même si on voit mieux en T2, RSI peut quantifier ce que T2 ne fait que montrer qualitativement
