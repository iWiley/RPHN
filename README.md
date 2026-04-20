# RPHN: Radiology-Pathology Hybrid Network

Cleaned repository for the Radiology-Pathology Hybrid Network (RPHN) project in
hepatocellular carcinoma, including the manuscript-facing training code,
runtime assets, and publication release materials.

## Layout

- `src/`: model, data pipeline, training, and evaluation utilities
- `configs/`: runnable experiment configs
- `assets/`: lightweight runtime assets
- `publication/`: released figures, tables, supplementary files, and release
  weights

## Quick Start

```bash
python -m src.train configs/default.yaml
```

The default configs expect local backbone weights to be available outside this
repository.

## Citation

Citation information for the manuscript will be updated upon publication.