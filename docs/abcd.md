# ABCD RSI Analysis Guide

## Overview

Restriction Spectrum Imaging (RSI) is a multi-shell diffusion MRI technique that decomposes the diffusion signal into cellular compartments. In ABCD, RSI provides voxel-wise and region-wise metrics across the whole brain.

### RSI Metrics Available in ABCD

| Metric | NDA Table | Description |
|--------|-----------|-------------|
| **RNI** (Restricted Normalized Isotropic) | `abcd_drsip101` | Isotropic restricted diffusion — cellularity marker |
| **RND** (Restricted Normalized Directional) | `abcd_drsip201` | Directional restricted diffusion — fiber integrity |
| **RNT** (Restricted Normalized Total) | `abcd_drsip301` | Total restricted signal (RNI + RND) |
| **HNI** (Hindered Normalized Isotropic) | `abcd_drsip401` | Hindered (extracellular) isotropic diffusion |
| **FNI** (Free Normalized Isotropic) | `abcd_drsip701` | Free water fraction (CSF-like) |

Each table contains ~224 variables covering:
- **White matter tracts** (AtlasTrack): fornix, cingulum, CST, ATR, uncinate, SLF, ILF, IFOF, corpus callosum (~44 vars)
- **Subcortical structures** (aseg): thalamus, caudate, putamen, hippocampus, amygdala, WM, ventricles (~28 vars)
- **Cortical regions** (Desikan atlas): peri-cortical WM, gray matter, gray-white contrast (~3×68 vars)
- **QC metrics**: `dmri_rsi_meanmotion`, `dmri_rsi_meantrans`, `dmri_rsi_meanrot`

---

## Paths on Wynton

```
/wynton/group/abcd/                         # Main ABCD data root (DataLoader)
├── 4.0/tabulated/img/                      # Imaging tabulated data (v4.0)
├── 5.0/tabulated/                          # v5.0 data
├── 5.1/tabulated/                          # v5.1 data
├── {version}/utils/                        # Variable lists, data dictionaries
│   ├── variable_list_v{version}.csv
│   └── data_dictionary_{version}.csv

/wynton/group/rsl/ABCD_data/                # RSL lab ABCD data
├── 4.0/imaging/concatenated/               # Concatenated imaging files
│   ├── imaging_qc.csv                      # QC data (fsqc_qc, iqc_dmri_ok_ser)
│   └── *.pkl                               # Tractometry pickles
├── labeled_data/
│   └── confounding_factors.csv

/mnt/fac/CX500002_DS1/lab-utils/            # Apptainer containers (RSI tools)
```

---

## abcd-utils

### Installation (on Wynton)

```bash
pip install git+ssh://git@github.com:rauschecker-sugrue-labs/abcd-utils.git
```

### Loading RSI Variables

```python
from abcd.data_loader import DataLoader

DL = DataLoader(version='5.1')  # or '5.0', '4.0'

# Search for RSI variables
rsi_vars = DL.find_variable('dmri_rsi')

# Search specific RSI metric
rni_vars = DL.find_variable('rsi_rni')   # Restricted Normalized Isotropic
rnd_vars = DL.find_variable('rsi_rnd')   # Restricted Normalized Directional

# Load a specific RSI variable
rni_data = DL.get_variable_data('dmri_rsi_meanmotion')

# Load multiple RSI variables
vars_list = ['dmri_rsi_meanmotion', 'dmri_rsi_meantrans']
df = DL.get_variables_data(vars_list)
```

### Combining RSI with Demographics & QC

```python
from abcd import utils as ut
from abcd.data_loader import DataLoader

DL = DataLoader(version='5.1')

# Get demographics
demos = DL.get_demographics()

# Get RSI data
rsi_rni = DL.get_variables_data(['dmri_rsi_rni_...'])  # fill with actual var names

# Merge
df = ut.abcd_merge([demos, rsi_rni])

# Apply QC filter (FreeSurfer + dMRI)
df_clean = ut.qc_filter(df, fsqc=True, dmri=True)
```

### Tractometry with RSI

```python
from abcd.tractutils import load_tractometry, merge_labels_features

# Load tractometry (DTI by default)
tracto = load_tractometry('tractometry_file_name')

# Available tracts (20 bundles):
# CST_R/L, UNC_R/L, IFO_R/L, ARC_R/L, ATR_R/L, CGC_R/L, HCC_R/L,
# FP (Forceps Major), FA (Forceps Minor), ILF_R/L, SLF_R/L
```

---

## RSI Processing with Apptainer

The RSI processing container is at:
```
/mnt/fac/CX500002_DS1/lab-utils/
```

### Running on Wynton (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=rsi_process
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rsi_%j.out

# Load apptainer module
module load apptainer

# Run RSI container
# Adapt paths and arguments based on container contents
apptainer run \
    --bind /wynton/group/abcd:/data \
    --bind /wynton/group/rsl:/output \
    /mnt/fac/CX500002_DS1/lab-utils/<container_name>.sif \
    <command> <args>
```

> **TODO**: Inspect the actual container at `/mnt/fac/CX500002_DS1/lab-utils/` to get:
> - Container filename (.sif)
> - Available commands (`apptainer inspect`, `apptainer run-help`)
> - Required inputs/outputs

---

## RSI for WMA/Leukoaraiosis Detection

RSI metrics relevant to white matter abnormalities:

| Metric | Relevance to WMA |
|--------|-------------------|
| **RNI** | Elevated in areas of increased cellularity (inflammation, gliosis) |
| **RND** | Reduced in demyelination/axonal damage — sensitive to WM integrity |
| **FNI** | Elevated in edema, CSF contamination — free water marker |
| **HNI** | Captures extracellular changes in WM lesions |
| **RNT** | Overall restricted signal — general tissue integrity |

### Analysis Ideas

1. **Region-based**: Compare RSI metrics (RNI, RND, FNI) in WMA-positive vs WMA-negative subjects across Desikan ROIs and WM tracts
2. **Tract-based**: Use AtlasTrack fiber tract RSI values to identify which tracts are most affected
3. **Voxel-wise**: Process raw RSI maps with the apptainer container for voxel-level analysis
4. **Longitudinal**: ABCD has Baseline, Year 2, Year 4 — track RSI changes over time in WMA subjects
5. **Multimodal**: Combine RSI metrics with the existing ResNet-3D classifier outputs (WMA predictions) as complementary features

---

## Quick Reference: NDA Variable Naming

RSI variable names in ABCD follow this pattern:
```
dmri_rsi{metric}_{atlas}_{region}_{hemisphere}
```

Examples:
- `dmri_rsi_rni_fib_at_fxcutl_lh` — RNI, fiber atlas, fornix cut left hemisphere
- `dmri_rsi_rnd_scs_cbwmtl_lh` — RND, subcortical seg, cerebral WM left
- `dmri_rsi_rni_dsk_pcwml_bkcdcdl_lh` — RNI, Desikan, peri-cortical WM left

Use `DL.find_variable('dmri_rsi_rni')` to explore all available variables interactively.

---

## Useful Commands (Wynton)

```bash
# SSH to Wynton
ssh <user>@log1.wynton.ucsf.edu

# Check available data versions
ls /wynton/group/abcd/

# Find RSI tabulated files
find /wynton/group/abcd/5.1/tabulated -name "*rsi*" -o -name "*drsip*"

# Inspect apptainer container
apptainer inspect /mnt/fac/CX500002_DS1/lab-utils/<container>.sif
apptainer run-help /mnt/fac/CX500002_DS1/lab-utils/<container>.sif

# Interactive shell in container
apptainer shell --bind /wynton/group/abcd:/data /mnt/fac/CX500002_DS1/lab-utils/<container>.sif

# Submit SLURM job
sbatch jobs/slurm_rsi.sh
```
