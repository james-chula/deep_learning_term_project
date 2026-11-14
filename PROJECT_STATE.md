# Project state

**Last updated:** 14 November 2026, 09:02 Asia/Bangkok
**Project:** BloodMNIST Deep Learning Term Project
**Execution status:** Stage-by-stage progress-report Colab workflow implemented and statically verified. The user supplied one completed validation result (accuracy 0.7383, macro-F1 0.6956, macro OvR ROC-AUC 0.9613; n=1,712), but its run provenance/artifact bundle still requires verification before formal report inclusion.

## Current approved implementation

A single validation-only procedural workflow supports:

- `majority` baseline;
- `linear_probe` (ImageNet-pretrained ResNet-18, frozen backbone, trainable final layer);
- `finetune` (ImageNet-pretrained ResNet-18, all layers trainable).

Active implementation is root `config.py`, `data.py`, `model.py`, `train.py`, and `run_experiment.py`, with one provider at `src/bloodmnist_baselines/data.py`. `CONFIG["mode"]` is the only workflow selector.

The canonical shared provider API is:

```python
build_provider_datasets(
    *, audit_manifest_path, image_size, train_transform, val_transform,
    subset_manifest_path=None,
) -> ProviderDatasets(train, val, metadata)
```

The provider is exported from `bloodmnist_baselines`; consumers use the returned
attributes (`datasets.train`, `datasets.val`, `datasets.metadata`), not dictionary
access. The Project Lead reported canonical code SHA
`8a637d34017bce6311225b82fd5d3a6563d7dd26`; this working-tree alignment has not
been an experiment and awaits Lead Git review.

## Safety contract

- Provider constructs official training and validation splits only; test data have no active loading path.
- Official 28×28 images are resized once to 224×224 in active transforms.
- Fixed 1% membership is `splits/bloodmnist_train_1pct_seed2026.npz`, seed 2026; it is train-only and failures do not fall back to full training.
- Validation does not update weights.
- Neural checkpoints are selected by validation macro-F1 only.
- Run evidence records `test_evaluated: false`.

## Git / Colab status

- Stabilization source commit pushed to `origin/main`: `cbfc3d741cb00eb9eccd568a09aa60e789f37646`.
- Thin launcher pin is `cbfc3d741cb00eb9eccd568a09aa60e789f37646`, the pushed source commit containing the reviewed provider, checkpoint-metrics, documentation, and safety changes.
- This follow-up pin commit documents the source pin; it does not change execution code beyond the launcher value.
- Repository remote: `git@github.com:james-chula/deep_learning_term_project.git`.
- Only Project Lead performs Git commits/pushes/pulls.

## Documentation and evidence

- Workflow: `SIMPLIFIED_WORKFLOW.md`.
- Decisions: `DECISION_LOG.md`.
- Future Colab evidence: `outputs/<timestamp>_<run-name>_<mode>/` with metadata, configuration, metrics, summary, and neural checkpoints/history/curve where applicable.

## Migration / removal

Removed as obsolete after active replacement: old separate baseline runner, metric module, baseline YAML config, duplicate/stale Colab notebooks, staged training framework, local data-audit notebook, old provider/interface documents, Nix/local Jupyter scaffolding, stale status/tracker/report templates, and generated caches. Required audit manifest, fixed subset manifest, proposal/schedule documents, source requirements, and active documentation are retained.

## Blocker and next action

**Deferred by user on 13 November 2026, 14:11 Asia/Bangkok:** controlled-matrix controls (augmentation toggle, LR-candidate/stage fields, seed-role fields, and expanded condition metadata) are deferred. They are required before controlled-matrix analysis, but are outside this stabilization/push.

**Blocker:** no Colab T4 execution authorization for the final Lead-reviewed pinned commit.
**Owner:** Project Lead for Git/pin; user for explicit execution approval.
**Next action:** Project Lead pushes the follow-up launcher-pin commit, then waits for explicit user authorization before any Colab T4 run. No Colab run is authorized by this Git action alone.


## 13 November 2026, 14:40 Asia/Bangkok — Colab audit-validation correction

The first Colab launch at source cbfc3d741cb00eb9eccd568a09aa60e789f37646 stopped before dataset construction because the audit artifact has a space before the class-3 parenthesis while MedMNIST 3.0.2 does not. Validation now canonicalizes only repeated whitespace and whitespace immediately before an opening parenthesis while retaining exact class-ID and semantic-label checks. Focused tests cover cosmetic whitespace, meaningful changes, swapped assignments, missing IDs, and non-string labels. Static suite: 10 tests passed. No dataset was loaded and no experiment ran during this correction. The corrected source commit is ba769bfb4aad80ad7ab0c0e54451b2516e99ab81; the launcher is pinned to it.

- Corrected source commit: ba769bfb4aad80ad7ab0c0e54451b2516e99ab81. The Colab launcher is pinned to this correction.

## 13 November 2026, 14:45 Asia/Bangkok — MedMNIST root creation correction

A Colab launch exposed that MedMNIST 3.0.2 requires its configured root directory to exist before dataset initialization. The provider now creates MEDMNIST_DATA_ROOT (default .data/medmnist) with parent directories before constructing train or validation datasets. A regression check enforces this ordering. No dataset or experiment was run for this correction. Corrected source commit: a7a4cd2ea490416374b95fb6301a129b5f190803. The Colab launcher is pinned to this source commit.

## 13 November 2026, 15:37 Asia/Bangkok — subset path source consolidated

The user pushed notebook entry-point update eb45400958b1ec185d02821155b75426f5ac239e. The fixed 1% subset path now comes from audit_manifest.json rather than a duplicate config.py value; any programmatic override must match. Static suite: 12 tests passed with PYTHONPATH=src. No dataset was loaded and no experiment ran. Source commit: 3e9d6ee25fcee2f26c5157568a5ced807da42b04. The Colab launcher is pinned to this source commit.

## 13 November 2026, 18:05 Asia/Bangkok — Colab checkout and data-root isolation

The Colab launcher now removes any stale /content/project directory before cloning and sets MEDMNIST_DATA_ROOT=/content/medmnist for the entry-point process. BloodMNIST data therefore resides outside the repository checkout under /content, while repository deletion/reclone does not target the dataset directory. Notebook structure and command assertions passed; no dataset was downloaded and no experiment ran. Source commit: b34cc5ed882b3d059d53465bbd9f72e8572af175. The launcher is pinned to this source commit; the following documentation-only commit publishes the pin.


## 13 November 2026, 20:40 Asia/Bangkok — stage-by-stage progress-report notebook

The Colab notebook now exposes runtime/provenance, configuration, data audit and samples, model/setup inspection, training, curves, explicit best-checkpoint reload and validation, confusion matrices, per-class metrics, misclassification examples, and evidence ZIP export as separate stages while reusing modular Python APIs. The CLI remains supported. Test loading remains absent. Adversarial review found no correctness blocker; 13/13 static unit tests passed and active Python compiled. Reviewed workflow source commit: 23f9f15f6020720b1320ba1c69dd8b0137eed814. The notebook is pinned to that source. Pin-publication commit: fe859e7. No experiment ran on the Team Computer.

## 13 November 2026, 21:13 Asia/Bangkok — concise plot label alias

Added `display_class_name()` to modular Python and updated the notebook sample visualization to use it. Class ID 3 is displayed as `immature granulocytes`; canonical audit/metric class names remain unchanged. Static suite: 14 tests passed; Python compilation and notebook JSON validation passed. No experiment ran on the Team Computer. Source commit: `6b71dff8ea41fb599cc33497de88c56700900881`; launcher pin commit: `f72c850d96c1f43197f548d41db3f4936b23306f`.


## 13 November 2026, 21:24 Asia/Bangkok — safe Colab reclone working directory

Added the Colab content-directory change before removing and cloning the project checkout, preventing stale-current-directory failures when the setup cell is rerun from inside the prior checkout. Notebook structure check and diff check passed. No experiment ran. Launcher fix commit: ff691e9c.


## 14 November 2026, 09:02 Asia/Bangkok — resilient Colab imports after reclone

Updated the Colab configuration cell to remove stale project paths, clear Python importer-cache entries for /content/project and /content/project/src, invalidate import caches, and discard a stale config module before import. The cell now also fails early if /content/project/config.py is absent. This addresses intermittent ModuleNotFoundError after deleting and recreating the checkout at the same path in a live Colab kernel. Notebook JSON validation and diff checks passed. No experiment ran. Expected commit: pending this Git operation.


## 14 November 2026, 10:51 Asia/Bangkok — identified multi-run Colab batch

The thin Colab notebook now accepts a validated `RUN_CONFIGS` list whose entries contain a unique notebook run ID and repository-config overrides. A fresh copy of repository defaults is applied for each run to prevent cross-run leakage. Notebook-only orchestration functions repeat the existing modular data/model/training/evaluation APIs sequentially. Per-run exceptions are recorded with the run ID, error type, message, and traceback; the batch continues with later runs and prints a final status list plus explicit failed IDs. Successful runs retain independent evidence ZIPs and are collected with `batch_results.json` in a batch ZIP. Duplicate IDs/run names, unknown config keys, and invalid modes fail before execution. Static suite: 14 tests passed; notebook JSON validation passed. No dataset or experiment was run on the Team Computer. Expected source commit: pending Git operation; launcher pin will be updated to the resulting source commit.
