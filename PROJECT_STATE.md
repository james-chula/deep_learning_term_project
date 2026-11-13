# Project state

**Last updated:** 13 November 2026, 14:11 Asia/Bangkok
**Project:** BloodMNIST Deep Learning Term Project
**Execution status:** No Colab experiment has run. No validation metrics, checkpoints, curves, or run evidence exist.

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
