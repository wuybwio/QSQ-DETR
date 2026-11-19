# QSD-DETR Experimental Artifacts

This repository contains the experimental result artifacts and plotting sources for the QSD-DETR paper on unknown-defect assessment in UAV power-line inspection.

## Contents

- `data/final_results/`: archived evaluation metrics, calibration artifacts, prediction summaries, and final-result plots copied from the RT-DETRv2 experiment output. JSON files were checked and machine-specific absolute paths were removed; relative sample identifiers and measured values are retained where present.
- `paper-experiments/figures/`: finalized experiment figures and available CSV source data, including vector PDF and PNG exports.
- `paper-experiments/scripts/`: scripts and notes used to generate or refine the paper experiment figures.
- `paper-experiments/experiments_final.tex`: the corresponding experiment-section source with the reported tables and discussion.

## Data Scope

The repository publishes the paper's derived experimental results and supporting figure data. It does not redistribute the InsPLAD images/annotations or detector training checkpoints. Obtain InsPLAD from its [original project page](https://andreluizbvs.github.io/InsPLAD/) and follow the dataset's stated license and attribution requirements. Some qualitative result figures contain derived visualizations based on InsPLAD examples; these are provided for research reference with attribution to the original dataset.

Large training outputs and intermediate/failed runs were not copied. The archived final evaluation folder includes the small calibration/evaluation artifacts needed to interpret the reported results. Paths specific to the original training machine were removed from structured JSON files.

## Reproducing Figures

The figure-generation scripts are in `paper-experiments/scripts/`. They were authored in the original experiment workspace and may expect the project dependencies and data files described in `README_figures.md`. Update any input/output directory arguments for your local environment before running them. Existing PDF/PNG exports and CSV plotting data are included so the published plots can be inspected without rerunning experiments.

## Provenance

- Final evaluation outputs: RT-DETRv2 `rtdetrv2_pytorch/output/final_results`.
- Figure bundle and plotting sources: finalized experiment bundle dated 2026-10-15.
- The original source data and checkpoints remain in the authors' experiment environment and are not included here.
