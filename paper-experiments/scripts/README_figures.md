# Experiment Figure Sources

This folder contains the plotting scripts for Figs. 6--12, the finalized figure bundle, its CSV plotting data, and figure-generation notes. The scripts were developed with Python, NumPy, Matplotlib, SciPy, and Pillow.

The scripts use command-line paths for the detector output tree, figure output directory, font folder, and (for detection examples) InsPLAD annotations and images. Supply paths for your own environment; the original machine-specific locations were removed. The InsPLAD dataset is not included in this repository.

Example command after installing dependencies and supplying local data paths:

```bash
python generate_experiment_figures.py \
  --repo-root /path/to/RT-DETRv2/rtdetrv2_pytorch \
  --figure-root /path/to/figure-output \
  --annotation /path/to/InsPLAD/annotations/instances_val_insplad17.json \
  --image-root /path/to/InsPLAD/val \
  --font-dir /path/to/Times-New-Roman-fonts \
  --dpi 450
```

The scripts may require per-image score and detector prediction files beyond the compact final-result archive in `data/final_results/`. Existing PDF/PNG exports and CSV plotting coordinates are included for inspection. `experiments_final.tex` records the matching experiment tables and discussion.

The CSV files are figure source data; the vector PDFs are the publication-ready exports. PNG files are raster previews and qualitative detection examples. Derived InsPLAD visualizations remain subject to the original dataset's attribution and non-commercial/share-alike terms.
