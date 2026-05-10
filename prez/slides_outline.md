# Slides — Détection de Leucoaraïose par Deep Learning
**ABCD Study | CHPC-UCSF | Mai 2026**

---

## Slide 1 — Titre

**Détection automatique de la leucoaraïose dans la cohorte ABCD**  
*Deep learning faiblement supervisé sur IRM T1w + T2w*

Louan Bardou  
Mai 2026 | CHPC-UCSF

---

## Slide 2 — Pourquoi ?

**La leucoaraïose chez l'adolescent : un signal précoce inexploré**

- Lésions de substance blanche détectables en IRM dès l'adolescence (~3–5% des scans ABCD)
- Chez l'adulte : prédicteur établi de démence et de déclin cognitif
- **Question :** Peut-on détecter ces lésions de façon automatique et systématique chez l'adolescent ?

**La cohorte ABCD : une opportunité unique**
- 11 900 sujets × 7 sessions = potentiellement 83 000 scans à analyser
- IRM T1w + T2w disponibles pour chaque visite
- Riche phénotypage cognitif, comportemental, génétique → analyses associatives futures

**Objectif à long terme :** Modèle spatio-temporel des trajectoires de la substance blanche

---

## Slide 3 — Trois défis fondamentaux

| Défi | Problème | Solution |
|------|----------|----------|
| **Pas de FLAIR** | Le standard clinique pour détecter les lésions n'est pas disponible dans ABCD | T1w + T2w en 2 canaux → le modèle apprend le contraste implicitement |
| **Pas de masks** | Aucune annotation voxel-à-voxel, seulement des labels image-niveau | Supervision faible + Grad-CAM pour localisation a posteriori |
| **Déséquilibre** | ~11% positifs → cross-entropy converge vers "tout négatif" | Optimiser l'AUPREC (Precision-Recall), insensible à la taille de la classe négative |

---

## Slide 4 — Architecture

**LeukoBinaryClassifier : Swin UNETR encodeur + tête GAP**

```
T1w + T2w (96³ voxels)
        ↓
Swin UNETR Encodeur     ← Vision Transformer 3D hiérarchique
  4 stages, attention fenêtrée (complexité linéaire)
        ↓
Feature map profonde    (B, 768, 3, 3, 3)
        ↓
Global Average Pool     → (B, 768)
        ↓
MLP head                → (B, 1) logit
```

**Pourquoi Swin Transformer ?**  
Les lésions diffuses couvrent de larges territoires → il faut du contexte global. Les CNNs classiques sont locaux.

**Pourquoi Global Average Pool ?**  
Force le réseau à activer fortement les régions lésées → heatmaps Grad-CAM exploitables en Phase 4.

**62.4M paramètres** | `use_checkpoint=True` (-40% VRAM)

---

## Slide 5 — Stratégie d'entraînement

**Pourquoi pas la cross-entropy standard ?**

| Loss | Problème |
|------|----------|
| Cross-entropy | Gradient dominé par les négatifs (88%) |
| AUROC | Triviallement haute si modèle prédit tout négatif |
| **AUPREC** ✓ | Mesure uniquement la capacité à ranker les positifs au-dessus des négatifs |

**APLoss (LibAUC)** : surrogate différentiable de l'AUPREC via ranking par paires  
**SOAP** : optimiseur couplé à APLoss, maintient une variable duale interne  
**DualSampler** : garantit ≥ 1 positif par batch (requis par APLoss)  
**StratifiedGroupKFold** (5 folds, groupes = subject_id) : évite la fuite de données entre sessions d'un même sujet

---

## Slide 6 — Pipeline de données : de 421 à 4466 samples

**Problème initial :** seulement 421 samples trouvés au lieu de ~4000 attendus

**Cause racine :** approche "labels-first" — on cherchait les fichiers à partir des CSVs de labels

```
Ancien pipeline :  CSV → chercher fichiers → 421 trouvés
Nouveau pipeline : Disque → chercher labels → 4466 trouvés
```

**Sources de labels fusionnées :**
- `labels.csv` : 1008 sujets sur disque (WMA + sains)
- `Baseline_healthy.csv` : 1709 sujets supplémentaires (sains, toutes sessions)
- `all_labels_merged.csv` : redondant pour les sujets sur disque

**Fix clé pour Baseline_healthy :** ce CSV ne couvrait que la session baseline (ses-00A), mais ces sujets ont des sessions de suivi sur disque → les traiter comme label=0 pour **toutes** leurs sessions.

**Résultat final :** 4466 lignes | 506 WMA (11.3%) | 3960 sains | 2708 sujets uniques

---

## Slide 7 — Problèmes techniques rencontrés

**6 bugs majeurs résolus :**

| # | Problème | Impact | Fix |
|---|----------|--------|-----|
| 1 | API LibAUC 1.3.0 (5 changements) | Training impossible | Inspecter localement avant déploiement |
| 2 | LR collapse à epoch 4 | AUPREC bloqué à 0.11 | Supprimer `update_regularizer`, LR 1e-4 → 1e-5 |
| 3 | `hidden_dim = feature_size × 32` | Crash initialisation | `hidden_dim = feature_size × 16 = 768` |
| 4 | 421 sujets au lieu de 4466 | Sous-utilisation des données | Approche disk-first |
| 5 | `ABCD_IMAGING` → mauvais path | 421 sujets sur cluster | Corriger `activate_env.sh` |
| 6 | I/O NFS : 15h pour epoch 1 | GPU à 0% d'utilisation | Cache tenseurs `.pt` sur scratch |

---

## Slide 8 — Résultats intermédiaires

**Run Fold 0 — 421 sujets (run initial, avant fix des données)**

| Métrique | Résultat | Baseline aléatoire |
|----------|----------|--------------------|
| Val AUPREC | **0.3613** | 0.12 |
| Val AUROC | **0.70** | 0.50 |
| Meilleure epoch | 94/100 | — |

**Interprétation :**
- AUPREC = 3× le baseline → détection réelle des cas positifs
- AUROC = 0.70 → bonne discrimination globale
- Modèle entraîné sur N=421 seulement → large marge de progression

**Run en cours :** Fold 0 avec 4466 samples → epoch 1 en cours (cache NFS en construction)

---

## Slide 9 — État actuel & prochaines étapes

**Status :**
- ✅ Architecture complète et validée
- ✅ Pipeline MONAI complet
- ✅ Manifest 4466 samples (2708 sujets)
- ✅ Cache I/O implémenté
- 🔄 Training Fold 0 (4466 samples) — epoch 1 en cours
- ⏳ Folds 1–4
- ⏳ Grad-CAM Phase 4 (si AUPREC > 0.50)
- ⏳ Segmentation Phase 5 (si AUPREC > 0.60)

**Prochaines étapes :**
1. Évaluer résultats Fold 0 avec 4466 samples
2. Cross-validation complète (5 folds)
3. Analyser biais de session (ses-02A/06A enrichis en WMA)
4. Grad-CAM → heatmaps de localisation des lésions
5. Analyses associatives (trajectoires longitudinales, corrélats cognitifs)
