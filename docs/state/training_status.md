# Training status

**Last updated:** 2026-11-13 10:58 Asia/Bangkok  
**Expected integration base:** stable base confirmed by Project Lead; execution SHA pending explicit authorization.

## Status
A complete staged validation-only ResNet-18 training implementation is prepared for Lead integration. It has **not** been integrated, committed, run locally, or run in Colab.

## Data contract
- Canonical archive/data audit remain provider-owned: 28px source, caller-supplied deterministic 224 resize applied by the canonical provider.
- Required paths after integration: `artifacts/data_audit/audit_manifest.json`, and authorized subset `splits/bloodmnist_train_1pct_seed2026.npz`.
- The canonical baseline module must export `build_provider_datasets(*, audit_manifest_path, image_size=224, train_transform, val_transform, subset_manifest_path=None)`.
- Provider results expose `.train`, `.val`, `.metadata`; training validates 11,959/1,712/3,421 split counts, 28px source, 224px resize, and `test_policy="locked_not_loaded"`. No test dataset is returned or evaluated.

## Implementation evidence
- Package: `staged_training_pipeline/src/bloodcell_training/`.
- Entry point: `python -m bloodcell_training.cli train --config ... --provider-module bloodmnist_baselines`.
- Thin notebook: `staged_training_pipeline/notebooks/colab_train_resnet18.ipynb`.
- Runtime outputs: `run_metadata.json`, `history.csv`, `learning_curves.png`, `best_validation_metrics.json`, `summary.json`, `best.pt`, `last.pt`, and append-only ledger JSONL.

## Next action
Provider module path corrected to bloodmnist_baselines after Lead static review. Lead integrates the bundle with the canonical provider function, then confirms a committed execution SHA and Colab pull before any run.
