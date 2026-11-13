# Training Status

**Last updated:** 2026-11-13 14:08 Asia/Bangkok
**Owner:** Training Engineer
**Git state:** Not inspected or changed in this update. Final reviewed commit and Colab pin are pending Project Lead review/commit.

## Current status

- **No experiment has run.** No Colab training, validation run, test evaluation, checkpoint generation, or Git operation was performed in this update.
- Training remains validation-only. The canonical provider is the sole source for training/validation data; test data remains locked and must not be loaded by the training package.

## Correctness fix staged in the canonical working tree

The neural training flow now retains the validation outputs from the epoch selected by highest validation macro-F1. `best_model.pt` is saved at that same selected epoch, and `metrics.json` and `summary.json` report metrics computed from those selected-epoch outputs—not from the last epoch. `last_model.pt` remains an atomic record of the final epoch, while epoch-by-epoch values remain in `history.csv`.

## Notebook pin status

`notebooks/colab_run.ipynb` now uses the fail-closed placeholder `PASTE_40_CHAR_REVIEWED_COMMIT_SHA` and validates the supplied value as a 40-character hexadecimal SHA before checkout. A final commit containing the reviewed changes and its corresponding Colab pin are still pending Project Lead action. The prior `ed0f975...` notebook pin is obsolete; it was not replaced with the current shared-base SHA because that base predates the working-tree fixes.

## Static verification completed

- Python compilation passed for the active training modules, including `config.py`, `data.py`, `model.py`, `train.py`, and `run_experiment.py`.
- `notebooks/colab_run.ipynb` passed JSON parsing.
- `PYTHONPATH=src python3 -m unittest tests/test_simple_workflow.py` passed: 5 tests, 0 failures.

## Next action / handoff

Project Lead: review the uncommitted working-tree changes, select a final commit, and update the notebook's reviewed SHA pin after that commit is available. Only after GitHub confirmation and a Colab pull of that exact commit may an authorized T4 validation run proceed.
