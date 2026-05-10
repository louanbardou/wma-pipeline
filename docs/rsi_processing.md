# RSI Processing Guide

## What it does

`RSI/RSIproc_1_0_8.py` decomposes multi-shell dMRI into 3 compartments via spherical harmonic fitting:

| Compartment | Isotropic (n0) | Directional (nd) | Total (nt) |
|---|---|---|---|
| **Restricted** (intracellular) | `n0s1.nii.gz` = **RNI** | `nds1.nii.gz` = **RND** | `nts1.nii.gz` = **RNT** |
| **Hindered** (extracellular) | `n0s2.nii.gz` = **HNI** | `nds2.nii.gz` = **HND** | `nts2.nii.gz` = **HNT** |
| **Free** (CSF) | `n0s3.nii.gz` = **FNI** | — (isotropic only) | `nts3.nii.gz` = **FNT** |

Also outputs `L2norm.nii.gz` (total signal norm).

## Inputs required

```
python RSIproc_1_0_8.py <dwi_eddy_corrected.nii.gz> <brain_mask.nii.gz> <file.bval> <file.bvec>
```

1. **dwi** — eddy-corrected 4D dMRI (multi-shell, includes b=0 volumes)
2. **mask** — binary brain mask in same space
3. **bval** — b-values file (space-separated, one per volume)
4. **bvec** — b-vectors file (3×N or N×3)

## Data location (CHPC)

```
/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-{ID}/ses-00A/dwi/
├── sub-{ID}_ses-00A_run-01_dwi.nii.gz     # raw dMRI (may need eddy correction)
├── sub-{ID}_ses-00A_run-01_dwi.bval       # check if present
└── sub-{ID}_ses-00A_run-01_dwi.bvec       # check if present
```

Test subjects with confirmed dMRI: `2HLV1V0P`, `2HLZM8RB`, `2JB8MUAJ`, `701F04JM`, `80VAXPN1`, `93JXKKF3`.

## Preprocessing before RSI

The script expects eddy-corrected data + a brain mask. If raw:

```bash
# 1. Extract brain mask (FSL)
bet <b0_volume> b0_brain -m -f 0.3

# 2. Eddy correction (FSL)
eddy --imain=dwi.nii.gz --mask=b0_brain_mask.nii.gz --bvals=dwi.bval --bvecs=dwi.bvec \
     --acqp=acqparams.txt --index=index.txt --out=dwi_eddy

# Or if mproc data is already corrected (likely — "mproc" = minimally processed):
# just extract the mask and run directly
```

## Run RSI (single subject)

```bash
cd $WMA_DIR
python rsi/RSIproc_1_0_8.py \
    /mnt/fac/.../sub-2HLV1V0P/ses-00A/dwi/sub-2HLV1V0P_ses-00A_run-01_dwi.nii.gz \
    b0_brain_mask.nii.gz \
    sub-2HLV1V0P_ses-00A_run-01_dwi.bval \
    sub-2HLV1V0P_ses-00A_run-01_dwi.bvec
# Outputs n0s1..3, nds1..2, nts1..3, L2norm.nii.gz in current directory
```

## Run RSI (batch, 6 subjects)

```bash
#!/bin/bash
MPROC=/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc
RSI_SCRIPT=$WMA_DIR/rsi/RSIproc_1_0_8.py
OUT_ROOT=/mnt/scratch/user/$USER/rsi_output

for SUBJ in 2HLV1V0P 2HLZM8RB 2JB8MUAJ 701F04JM 80VAXPN1 93JXKKF3; do
    DWI_DIR=$MPROC/sub-$SUBJ/ses-00A/dwi
    DWI=$(ls $DWI_DIR/*_dwi.nii.gz | head -1)
    BVAL=$(ls $DWI_DIR/*_dwi.bval | head -1)
    BVEC=$(ls $DWI_DIR/*_dwi.bvec | head -1)

    OUT=$OUT_ROOT/sub-$SUBJ
    mkdir -p $OUT && cd $OUT

    # Extract b0 and create mask
    fslroi $DWI b0 0 1
    bet b0 b0_brain -m -f 0.3

    python $RSI_SCRIPT $DWI b0_brain_mask.nii.gz $BVAL $BVEC
    echo "Done: $SUBJ"
done
```

## SLURM job

```bash
#!/bin/bash
#SBATCH --job-name=rsi_proc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rsi_%j.out

module load fsl
source $WMA_DIR/activate_env.sh
# run batch script above
```

No GPU needed — RSI is pure NumPy/SciPy (CPU only, ~10-30 min per subject).

## Visualize RSI maps

### Python (nilearn)

```python
import nilearn.plotting as plotting

# Single subject RSI map
plotting.plot_anat('n0s1.nii.gz', title='RNI (Restricted Isotropic)')
plotting.plot_anat('nds1.nii.gz', title='RND (Restricted Directional)')
plotting.plot_anat('n0s3.nii.gz', title='FNI (Free Water)')
plotting.show()

# Overlay RSI on T1w
plotting.plot_anat('T1w.nii.gz', title='RNI overlay')
plotting.plot_stat_map('n0s1.nii.gz', bg_img='T1w.nii.gz', threshold=0.1,
                       title='RNI on T1w', cmap='hot')
plotting.show()

# Compare WMA+ vs WMA- (after computing group maps)
plotting.plot_glass_brain('rni_tstat.nii.gz', title='RNI: WMA+ vs WMA-', threshold=2.0)
```

### Interactive (fsleyes — recommended for detailed inspection)

```bash
pip install fsleyes
fsleyes T1w.nii.gz n0s1.nii.gz -cm hot -a 50 nds1.nii.gz -cm blue-lightblue -a 50
```

### ITK-SNAP (GUI, if available)

```bash
itksnap -g T1w.nii.gz -o n0s1.nii.gz
```

## WMA relevance

- **RNI elevated** in WMA regions = gliosis/inflammation (cellular proliferation)
- **RND reduced** in WMA = demyelination/axonal loss
- **FNI elevated** in WMA = edema, tissue breakdown
- Can be added as extra input channels to ResNet-3D (T1w + T2w + RNI + RND + FNI = 5 channels)
