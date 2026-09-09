# Historical reproduction

Full pre-restructure checkpoint: [archive/2026_09_09_10_30_34_before_unified_layout](https://github.com/souray0410/LOOK/tree/archive/2026_09_09_10_30_34_before_unified_layout), Git commit `1d08c5890fb0325266907ac5e8bd34c7cc8c8f77`. Check out this branch in a separate directory and use its recorded dependencies and source/data manifests to reproduce historical experiments. Its reports/configuration/acceptance records retain original bytes.

Current scientific raw/checkpoint storage remains where recorded; moving code in the new branch does not delete or rewrite those artifacts. Old Python object serialization may require the old import paths/environment. New work must not silently treat old test data as a fresh tuning set.
