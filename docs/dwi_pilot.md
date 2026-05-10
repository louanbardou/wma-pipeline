# DWI/RSI Pilot — WMA Detectability Test

> Objectif: tester si les maps RSI permettent de distinguer les sujets WMA+ des sujets sains au baseline.

## Data source

All data is on CHPC at:
```
/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/
```

Structure per subject:
```
sub-XXXXXXXX/
├── ses-00A/
│   ├── anat/
│   │   ├── sub-XXXXXXXX_ses-00A_run-01_T1w.nii.gz
│   │   └── sub-XXXXXXXX_ses-00A_run-01_T2w.nii.gz
│   └── dwi/     ← only present for ~12 subjects in mproc
│       └── ...
```

**Important**: only 12 sessions in all of `mproc` have a `dwi/` folder. For RSI processing at scale, use tabulated data on Wynton (`abcd-utils` DataLoader) or process raw dMRI with the `pyrsi` apptainer container.

---

## Pilot subjects (all baseline, ses-00A)

### WMA+ subjects (6)

**NDAR_INV9PJ7VRDA** (sub-9PJ7VRDA) — mrif_score=3
> Motion degraded T2 images, with more supratentorial white matter T2 hyperintensities (better seen at T1 hypointensities, as T1 images are less motion degraded) than usually seen for age. Finding is nonspecific and may represent the sequela of prior injury, ischemia, demyelination or inflammation. Recommend correlation with patient symptoms and clinical MRI or referral as indicated.

**NDAR_INV61PF7E1L** (sub-61PF7E1L) — mrif_score=3
> Motion degraded T2 images. 5mm ovoid T2 hyperintense T1 hypointense lesion in the lateral aspect of the left medulla, in the general region of the trigeminal nucleus and tract. This finding is non-specific and may reflect sequela of remote infectious, inflammatory, or ischemic injury. An active inflammatory process could also have this appearance. No significant mass effect to suggest a neoplastic process. Recommend correlation with clinical history and consideration of dedicated clinical brain MRI.

**NDAR_INV80VAXPN1** (sub-80VAXPN1) — mrif_score=3
> Significant T2 hyperintensity in the white matter adjacent to bilateral frontal horns, nonspecific, may represent the sequela of prior injury or a dysplasia. Recommend correlation with clinical history and consideration of clinical MRI.

**NDAR_INV93JXKKF3** (sub-93JXKKF3) — mrif_score=3
> Single area of T2 signal hyperintensity in the left frontal periventricular white matter is non specific and may reflect sequela of remote infectious, inflammatory, or ischemic injury. Recommend correlation with any recent history of neurological symptoms that might suggest an active demyelinating disorder.

**NDAR_INV99TVX9G8** (sub-99TVX9G8) — mrif_score=3
> Multiple supratentorial white matter T2 hyperintensities. These are non-specific and may reflect sequela of remote infectious, inflammatory, or ischemic injury. Recommend correlation with clinical history.

**NDAR_INV701F04JM** (sub-701F04JM) — mrif_score=3
> Multiple white matter T2 hyperintensities some with a more confluent appearance located predominantly within the frontal and parietal lobes and primarily involving the subcortical white matter but sparing the immediate juxtacortical U fibers. These may reflect sequela of remote infectious, inflammatory, or ischemic injury - however, the more confluent lesions and the pattern sparing the U fibers is concerning for early manifestation of a primary leukoencephalopathy such as metachromatic leukodystrophy. Recommend correlation with clinical history. If the patient is asymptomatic recommend consideration of follow up imaging in 6 months to document stability of these findings which would support the diagnosis of sequela of remote injury over an active process.

### Healthy subjects (6)

**NDAR_INV2HLV1V0P** (sub-2HLV1V0P) — mrif_score=1
> Motion degraded T2 images, no large abnormality

**NDAR_INV2HLV10CC** (sub-2HLV10CC) — mrif_score=1
> Motion degraded - no large abnormality

**NDAR_INV2HLZM8RB** (sub-2HLZM8RB) — mrif_score=1
> Motion degraded T2 images, no large abnormality

**NDAR_INV2J3D85NJ** (sub-2J3D85NJ) — mrif_score=1
> Mildly motion degraded images, no large abnormality

**NDAR_INV2JB8MUAJ** (sub-2JB8MUAJ) — mrif_score=1
> Motion degraded T2 images, no large abnormality

**NDAR_INV2K3JH38W** (sub-2K3JH38W) — mrif_score=1
> T2 images are motion degraded

---

## Full paths (baseline ses-00A)

### WMA+ subjects

```bash
# NDAR_INV9PJ7VRDA (sub-9PJ7VRDA)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-9PJ7VRDA/ses-00A/anat/sub-9PJ7VRDA_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-9PJ7VRDA/ses-00A/anat/sub-9PJ7VRDA_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-9PJ7VRDA/ses-00A/dwi/

# NDAR_INV61PF7E1L (sub-61PF7E1L)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-61PF7E1L/ses-00A/anat/sub-61PF7E1L_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-61PF7E1L/ses-00A/anat/sub-61PF7E1L_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-61PF7E1L/ses-00A/dwi/

# NDAR_INV80VAXPN1 (sub-80VAXPN1)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-80VAXPN1/ses-00A/anat/sub-80VAXPN1_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-80VAXPN1/ses-00A/anat/sub-80VAXPN1_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-80VAXPN1/ses-00A/dwi/

# NDAR_INV93JXKKF3 (sub-93JXKKF3)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-93JXKKF3/ses-00A/anat/sub-93JXKKF3_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-93JXKKF3/ses-00A/anat/sub-93JXKKF3_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-93JXKKF3/ses-00A/dwi/

# NDAR_INV99TVX9G8 (sub-99TVX9G8)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-99TVX9G8/ses-00A/anat/sub-99TVX9G8_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-99TVX9G8/ses-00A/anat/sub-99TVX9G8_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-99TVX9G8/ses-00A/dwi/

# NDAR_INV701F04JM (sub-701F04JM)
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-701F04JM/ses-00A/anat/sub-701F04JM_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-701F04JM/ses-00A/anat/sub-701F04JM_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-701F04JM/ses-00A/dwi/
```

### Healthy subjects

```bash
# NDAR_INV2HLV1V0P (sub-2HLV1V0P) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV1V0P/ses-00A/anat/sub-2HLV1V0P_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV1V0P/ses-00A/anat/sub-2HLV1V0P_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV1V0P/ses-00A/dwi/

# NDAR_INV2HLV10CC (sub-2HLV10CC) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV10CC/ses-00A/anat/sub-2HLV10CC_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV10CC/ses-00A/anat/sub-2HLV10CC_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLV10CC/ses-00A/dwi/

# NDAR_INV2HLZM8RB (sub-2HLZM8RB) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLZM8RB/ses-00A/anat/sub-2HLZM8RB_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLZM8RB/ses-00A/anat/sub-2HLZM8RB_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2HLZM8RB/ses-00A/dwi/

# NDAR_INV2J3D85NJ (sub-2J3D85NJ) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2J3D85NJ/ses-00A/anat/sub-2J3D85NJ_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2J3D85NJ/ses-00A/anat/sub-2J3D85NJ_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2J3D85NJ/ses-00A/dwi/

# NDAR_INV2JB8MUAJ (sub-2JB8MUAJ) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2JB8MUAJ/ses-00A/anat/sub-2JB8MUAJ_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2JB8MUAJ/ses-00A/anat/sub-2JB8MUAJ_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2JB8MUAJ/ses-00A/dwi/

# NDAR_INV2K3JH38W (sub-2K3JH38W) — mrif_score=1, no findings
T1w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2K3JH38W/ses-00A/anat/sub-2K3JH38W_ses-00A_run-01_T1w.nii.gz
T2w: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2K3JH38W/ses-00A/anat/sub-2K3JH38W_ses-00A_run-01_T2w.nii.gz
dwi: /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/sub-2K3JH38W/ses-00A/dwi/
```

---

## Notes

- The `dwi/` folder may not exist for all subjects listed above. Only 12 sessions total in `mproc` have `dwi/`. Verify existence on CHPC before processing:
  ```bash
  for sub in sub-9PJ7VRDA sub-61PF7E1L sub-80VAXPN1 sub-93JXKKF3 sub-99TVX9G8 sub-701F04JM sub-2HLV1V0P sub-2HLV10CC sub-2HLZM8RB sub-2J3D85NJ sub-2JB8MUAJ sub-2K3JH38W; do
      echo -n "$sub: "
      ls /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc/$sub/ses-00A/ 2>/dev/null || echo "NOT FOUND"
  done
  ```
- All 6 WMA+ subjects have mrif_score=3 (finding needs follow-up) — these are not subtle cases
- All 6 healthy subjects have mrif_score=1 (normal) with no findings in any source CSV
- Healthy subjects are NOT in the `available_subjects.txt` / scratch download — they are only on `/mnt/fac/` storage
- For RSI maps, will need to process raw dMRI with `pyrsi` or use tabulated RSI from Wynton
