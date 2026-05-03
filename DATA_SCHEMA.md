# Data schema

This document describes the canonical data interface expected by the RPHN training and evaluation code.

RPHN uses paired arterial-phase CT, whole-slide histopathology patches, CT region masks, and patient-level OS/TTR outcomes. The repository does not define a universal hospital data-ingestion pipeline; users should convert compatible local data into the structure below.

## Cohort layout

Each cohort is represented by one HDF5 file and one survival CSV file:

```text
data/
  A/
    data.h5
    surv.csv
  B/
    data.h5
    surv.csv
  C/
    data.h5
    surv.csv
```

The split assignment is controlled by the YAML config, for example:

```yaml
train:
  data_h5: "data/A/data.h5"
  surv_csv: "data/A/surv.csv"

val:
  data_h5: "data/B/data.h5"
  surv_csv: "data/B/surv.csv"

test:
  - name: "C"
    data_h5: "data/C/data.h5"
    surv_csv: "data/C/surv.csv"
```

## HDF5 structure

Each `data.h5` file contains one top-level group per patient. The group name must match the patient ID used in `surv.csv`.

```text
data.h5
├── <PatientID>/
│   ├── wsi/
│   │   └── patches/
│   │       ├── images
│   │       └── coords
│   └── ct/
│       ├── bundle          # or slices
│       ├── mask/
│       │   ├── liver
│       │   ├── liver_lesion_or_tumor
│       │   ├── liver_peritumoral
│       │   └── liver_vessels
│       └── attrs
└── ...
```

Every patient group must contain both WSI and CT inputs.

## WSI fields

### `wsi/patches/images`

A sequence of encoded RGB pathology patch images.

The current loader expects JPEG XL-encoded patch bytes. After decoding, patches are converted to tensors and normalized for the GigaPath encoder.

Expected decoded patch shape:

```text
N × 3 × H × W
```

The encoder resizes patches internally to:

```text
224 × 224
```

### `wsi/patches/coords`

Patch coordinates with shape:

```text
N × 2
```

Coordinates are used for spatial positional encoding and graph construction. They should be numeric and should use a consistent slide coordinate system within each WSI.

## CT fields

Each patient group must contain either:

```text
ct/bundle
```

or:

```text
ct/slices
```

The preprocessing code converts CT data to numeric HU-space representation, applies intensity clipping/normalization, crops around the union of ROI masks with a 10 mm physical margin, and resizes the volume to:

```text
1 × 128 × 128 × 128
```

Required CT metadata:

```text
ct.attrs["zooms"]
```

For slice-based storage, the code also expects:

```text
ct.attrs["shape_xyz"]
```

Optional slice-reconstruction attributes include:

```text
ct.attrs["offset_for_uint16"]
ct.attrs["source_type"]
ct.attrs["slope"]
ct.attrs["inter"]
```

## CT ROI masks

Each patient must provide four CT ROI masks:

```text
ct/mask/liver
ct/mask/liver_lesion_or_tumor
ct/mask/liver_peritumoral
ct/mask/liver_vessels
```

Channel order after preprocessing:

```text
0: liver
1: liver_lesion_or_tumor
2: liver_peritumoral
3: liver_vessels
```

Each mask should be a 3D binary or binary-like array aligned to the source CT volume.

The model uses these masks for the explicit CT evidence stream. In the manuscript pipeline, masks were initialized by a liver-focused segmentation workflow and then physician-refined before consensus acceptance.

## Survival CSV

Each cohort requires a `surv.csv` file. The CSV index must be the patient ID and must match the HDF5 top-level group names.

Required columns:

```text
OS_Event
OS_Time
TTR_Event
TTR_Time
```

| Column | Meaning |
|---|---|
| `OS_Event` | Overall survival event indicator. `1` = death/event, `0` = censored. |
| `OS_Time` | Observed OS follow-up time in days. |
| `TTR_Event` | Time-to-recurrence event indicator. `1` = recurrence/event, `0` = censored. |
| `TTR_Time` | Observed TTR follow-up time in days. |

Example:

```csv
PatientID,OS_Event,OS_Time,TTR_Event,TTR_Time
A0001,0,2650,0,2650
A0002,0,2304,1,721
A0003,1,1040,1,752
```

All four required columns must be present and numeric.

## Optional clinical annotation

The training code only requires the survival CSV columns above.

Additional clinicopathologic variables used for manuscript-level statistical analysis may be stored separately, for example:

```text
publication/supplementary/table_s1_anonymized_patient_level_table.csv
```

These variables are not required by `src.train` unless a downstream analysis script explicitly uses them.

## Runtime assets

The data schema does not include third-party foundation-model weights.

The active code expects local backbone weights at the paths resolved in `src/eval_utils.py`, including:

```text
model/prov-gigapath
model/ct-fm/ct_fm_feature_extractor
```

The WSI concept stream also expects the anchor payload specified in the config, by default:

```text
assets/anchors/anchors_wsi.pth
```

## Validation checklist

Before running training or evaluation, check that:

1. Every HDF5 patient group has a matching row in `surv.csv`.
2. Each patient contains WSI patches, WSI coordinates, CT volume data, and all four CT ROI masks.
3. `surv.csv` contains `OS_Event`, `OS_Time`, `TTR_Event`, and `TTR_Time`.
4. Survival times are numeric and expressed in days.
5. Event indicators use `1` for event and `0` for censoring.
6. CT masks are aligned with the source CT before preprocessing.
7. WSI coordinates have shape `N × 2`.
8. Required local backbone weights and WSI anchors are available.
