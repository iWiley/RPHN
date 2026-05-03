# RPHN: Radiology-Pathology Hybrid Network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19659612.svg)](...)

Cleaned repository for the Radiology-Pathology Hybrid Network (RPHN) project in
hepatocellular carcinoma, including the manuscript-facing training code,
runtime assets, and publication release materials.

## Layout

- `src/`: model, data pipeline, training, and evaluation utilities
- `configs/`: runnable experiment configs
- `assets/`: lightweight runtime assets
- `publication/`: released figures, tables, supplementary files, and release
  weights

## Release scope and data availability

This repository contains the manuscript-facing implementation of RPHN, including core code, configuration files, runtime assets, and released model artifacts.

Raw institutional CT images, whole-slide images, annotations, and patient-level clinical materials are not distributed because of patient privacy, ethical approval, institutional governance, and data-use restrictions.

RPHN uses external foundation-model backbones. Third-party backbone weights are not redistributed in this repository; users should obtain them from the original sources and comply with their respective licenses and access terms.

The public release is intended for code inspection, methodological review, and use with compatible data under the stated license. Full reproduction from institutional raw source data is subject to the data-access constraints described in the manuscript.

## Runtime assets

The default training code expects local third-party backbone weights at:

```text
model/prov-gigapath
model/ct-fm/ct_fm_feature_extractor
```

RPHN uses a project-specific WSI anchor bank for the explicit pathology concept stream. The released anchor payload is provided at:

```text
assets/anchors/anchors_wsi.pth
```

Anchor packing utilities are available under `src/anchors/`. For the expected HDF5/CSV input structure, see `DATA_SCHEMA.md`.

## Quick Start

```bash
python -m src.train configs/default.yaml
```

The default configs expect local backbone weights to be available outside this
repository.

## License and citation

This repository is released for non-commercial academic use under the CC BY-NC 4.0 license.

If you use this code, please cite the associated manuscript. A `CITATION.cff` file will be updated after publication.