# Data schema

This document summarizes the HDF5 input structure expected by the RPHN data loader. It is intended as a compact interface note rather than a full data-availability statement.

## Cohort files

Each cohort is configured with one HDF5 file and one survival CSV file:

```yaml
train:
  data_h5: "data/A/data.h5"
  surv_csv: "data/A/surv.csv"
```

The HDF5 file contains the paired CT and WSI inputs. The CSV only provides patient-level survival labels used by the Cox loss.

## HDF5 structure

Each `data.h5` file contains one top-level group per patient. The group name is the patient ID and should match the corresponding row in `surv.csv`.

```text
data.h5
├── <PatientID>/
│   ├── wsi/
│   │   └── patches/
│   │       ├── images
│   │       └── coords
│   └── ct/
│       ├── bundle          # preferred compact CT storage
│       ├── slices          # alternative slice-based CT storage
│       ├── mask/
│       │   ├── liver
│       │   ├── liver_lesion_or_tumor
│       │   ├── liver_peritumoral
│       │   └── liver_vessels
│       └── attrs
└── ...
```

Each patient group must contain one WSI patch set, one CT volume source, and the four CT ROI masks.

## WSI fields

### `wsi/patches/images`

Encoded RGB pathology patch images. The current loader expects JPEG XL-encoded patch bytes and decodes them with `imagecodecs`.

After decoding, patches are represented as:

```text
N × 3 × H × W
```

The GigaPath wrapper resizes and normalizes patches internally to the expected 224 × 224 ImageNet-normalized input.

### `wsi/patches/coords`

Patch coordinates with shape:

```text
N × 2
```

Coordinates are used for spatial positional encoding and graph construction, so they should use a consistent coordinate system within each slide.

## CT fields

Each patient group must contain either `ct/bundle` or `ct/slices`.

The preprocessing code reconstructs the CT volume, clips and normalizes intensity, crops around the union of ROI masks with a 10 mm physical margin, and resizes the result to:

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

Optional slice-reconstruction attributes:

```text
ct.attrs["offset_for_uint16"]
ct.attrs["source_type"]
ct.attrs["slope"]
ct.attrs["inter"]
```

## CT ROI masks

Required mask datasets:

```text
ct/mask/liver
ct/mask/liver_lesion_or_tumor
ct/mask/liver_peritumoral
ct/mask/liver_vessels
```

The loader stacks them in this order:

```text
0: liver
1: liver_lesion_or_tumor
2: liver_peritumoral
3: liver_vessels
```

Each mask should be a 3D binary or binary-like array aligned to the source CT volume. During preprocessing, masks are cropped with the CT volume and resized to:

```text
4 × 128 × 128 × 128
```

## Survival CSV

The CSV is intentionally minimal. Its index must match the HDF5 patient IDs. The required columns are:

```text
OS_Event
OS_Time
TTR_Event
TTR_Time
```

Times are expected in days; event indicators use `1` for event and `0` for censoring.

## Minimal check

Before training or evaluation, verify that every patient has:

- a matching HDF5 group and CSV row
- `wsi/patches/images`
- `wsi/patches/coords`
- `ct/bundle` or `ct/slices`
- all four CT masks listed above
- numeric OS and TTR labels in the CSV
