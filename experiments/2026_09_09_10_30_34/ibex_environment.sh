#!/bin/bash
# Source this file after activating this release's independent Python environment.
LOOK_IBEX_CODE=/ibex/project/c2377/souray/home/mengh/LOOK/2026_09_09_10_30_34
LOOK_IBEX_DATA=/ibex/project/c2377/souray/data/mengh/LOOK
export LOOK_PROJECT_ROOT="$LOOK_IBEX_CODE"
export LOOK_DATA_ROOT="$LOOK_IBEX_DATA"
export LOOK_RUNS_ROOT="$LOOK_IBEX_DATA/runs/2026_09_09_10_30_34"
export LOOK_DATASET_ROOT="$LOOK_IBEX_DATA/dataset/2026_09_09_10_30_34"
export LOOK_IMAGE_ROOT="$LOOK_DATASET_ROOT/images"
export LOOK_COHORT_ROOT="$LOOK_DATASET_ROOT/cohorts/pending_protocol"
export LOOK_LABELS_CSV="$LOOK_COHORT_ROOT/primary/reference_labels.csv"
export LOOK_NATURAL_LABELS_CSV="$LOOK_COHORT_ROOT/not_authorized/reference_labels.csv"
export LOOK_PREPROCESS_CACHE_ROOT="$LOOK_IBEX_DATA/cache/2026_09_09_10_30_34/preprocessed_pairs"
export LOOK_CACHE_ROOT="$LOOK_IBEX_DATA/cache/2026_09_09_10_30_34"
# Do not override Slurm CUDA_VISIBLE_DEVICES or reuse ws02 physical GPU indices.
