# Project state

**Last updated:** 13 November 2026, 13:19 Asia/Bangkok
**Project:** BloodMNIST Deep Learning Term Project
**Execution status:** No Colab experiment has run. No validation metrics, checkpoints, curves, or run evidence exist.

## Current approved implementation

A single validation-only procedural workflow supports:

- `majority` baseline;
- `linear_probe` (ImageNet-pretrained ResNet-18, frozen backbone, trainable final layer);
- `finetune` (ImageNet-pretrained ResNet-18, all layers trainable).

Active implementation is root `config.py`, `data.py`, `model.py`, `train.py`, and `run_experiment.py`, with one provider at `src/bloodmnist_baselines/data.py`. `CONFIG["mode"]` is the only workflow selector.

## Safety contract

- Provider constructs official training and validation splits only; test data have no active loading path.
- Official 28×28 images are resized once to 224×224 in active transforms.
- Fixed 1% membership is `splits/bloodmnist_train_1pct_seed2026.npz`, seed 2026; it is train-only and failures do not fall back to full training.
- Validation does not update weights.
- Neural checkpoints are selected by validation macro-F1 only.
- Run evidence records `test_evaluated: false`.

## Git / Colab status

- Base commit before consolidation: `49c9be4692088609611717a028cf5e845095d1d7` on `main`.
- Consolidation commit: ed0f97568df549986ddf2aecf11dee3548135ca4 (pending push).
- Thin launcher pin: ed0f97568df549986ddf2aecf11dee3548135ca4; do not execute an earlier pin.
- Repository remote: `git@github.com:james-chula/deep_learning_term_project.git`.
- Only Project Lead performs Git commits/pushes/pulls.

## Documentation and evidence

- Workflow: `SIMPLIFIED_WORKFLOW.md`.
- Decisions: `DECISION_LOG.md`.
- Future Colab evidence: `outputs/<timestamp>_<run-name>_<mode>/` with metadata, configuration, metrics, summary, and neural checkpoints/history/curve where applicable.

## Migration / removal

Removed as obsolete after active replacement: old separate baseline runner, metric module, baseline YAML config, duplicate/stale Colab notebooks, staged training framework, local data-audit notebook, old provider/interface documents, Nix/local Jupyter scaffolding, stale status/tracker/report templates, and generated caches. Required audit manifest, fixed subset manifest, proposal/schedule documents, source requirements, and active documentation are retained.

## Blocker and next action

**Blocker:** no Colab T4 execution authorization for this new consolidated pinned commit.
**Owner:** Project Lead for Git/pin; user for explicit execution approval.
**Next action:** push the consolidation commit, confirm origin/main, then review that exact pin before authorizing any Colab run.
