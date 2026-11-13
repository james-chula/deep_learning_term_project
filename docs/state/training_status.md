# Training Status

**Last updated:** 2026-11-13 20:35 Asia/Bangkok
**Owner:** Training Engineer
**Git state:** No Git commands or Git-state changes were performed. Final reviewed commit and Colab pin remain pending Project Lead review/commit.

## Current status

- **No experiment has run.** No Colab training, validation run, test evaluation, or result claim was produced.
- The canonical working tree now contains a staged Colab interface that calls the existing modular Python APIs for runtime/provenance, configuration, audited data/sample display, model summary, setup, training, curves, selected-checkpoint validation, diagnostics, and artifact ZIP creation.
- The command-line entry point remains `PYTHONPATH=src python run_experiment.py`.
- Training and notebook flows remain validation-only. The canonical provider is the sole source for training/validation data; test data stays locked and is never loaded.

## Selected-checkpoint evidence behavior

Neural training selects `best_model.pt` by validation macro-F1. Final validation metrics and diagnostic figures are now generated only after reloading that selected checkpoint. `last_model.pt` remains an atomic final-epoch record and `history.csv` remains the epoch-level record; neither is presented as the selected result.

## Colab notebook interface

`notebooks/colab_run.ipynb` uses the fail-closed `PASTE_40_CHAR_REVIEWED_COMMIT_SHA` placeholder and verifies the exact checked-out SHA. Its stages are:

1. T4 runtime and source provenance.
2. Plain-dictionary configuration.
3. Canonical-provider audit/count display and training-sample visualization.
4. Shared ResNet-18 model summary/trainable-parameter count (or majority-mode explanation).
5. Evidence-directory and provenance setup.
6. Majority evaluation or neural training through shared APIs.
7. Saved learning-curve display.
8. Explicit reload and validation of the macro-F1-selected checkpoint.
9. Count/normalized confusion matrices, per-class precision/recall/F1 chart, and misclassified validation samples.
10. Final metrics/summary and downloadable evidence ZIP.

The notebook contains orchestration and visualization calls only; loaders, model construction, optimization, metrics, checkpoint evaluation, and diagnostic generation remain in modular `.py` files.

## Static verification completed

- Python compilation passed for `config.py`, `data.py`, `model.py`, `train.py`, `run_experiment.py`, `src/bloodmnist_baselines/data.py`, and `tests/test_simple_workflow.py`.
- Notebook JSON and Python-cell syntax validation passed (Colab shell/magic lines excluded from Python AST parsing).
- `PYTHONPATH=src python3 -m unittest tests/test_simple_workflow.py` passed: 13 tests, 0 failures.

## Assumptions / pending action

- Colab supplies CUDA-compatible PyTorch/TorchVision; `requirements-colab.txt` supplies the other pinned packages.
- Validation loader ordering remains deterministic (`shuffle=False`), allowing saved predictions to align with displayed misclassified samples.
- Diagnostic class names come from the audited manifest, not a notebook-local list.
- Project Lead must review the working-tree changes, create/select the pushed commit, replace the notebook placeholder with that reviewed full SHA, and confirm the exact commit in Colab before any authorized T4 run.
