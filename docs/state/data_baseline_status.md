# Data & Baseline Engineering Status

**Last updated:** 2026-11-13 10:24 Asia/Bangkok
**Repository integration target:** `/home/rakazo/deep_learning_term_project`, `main` integration pending; Project Lead designates the post-integration commit.
**Git authority:** Project Lead only. This staged directory contains uncommitted proposed artifacts; this engineer has performed no Git action.

## Scope and current status

| Deliverable | Status | Evidence / intended path |
| --- | --- | --- |
| Dataset audit and class mapping | Verified | `docs/data_and_baselines.md` (staged) |
| Fixed official split protocol | Verified | `src/bloodmnist_baselines/data.py` |
| Fixed 1% subset manifest | Generated locally but not yet staged for integration | `splits/bloodmnist_train_1pct_seed2026.npz` |
| Majority-class validation baseline | Implementation staged; metrics pending Colab T4 | `src/bloodmnist_baselines/baselines.py` |
| Frozen ResNet-18 linear-probe validation baseline | Implementation staged; metrics pending Colab T4 | `src/bloodmnist_baselines/baselines.py` |
| Validation results and runtime evidence | Blocked pending integrated commit and Colab run | `artifacts/baseline_results/` |

## Decisions

1. Dataset identifier is `bloodmnist` in MedMNIST v2, multiclass RGB. Models consume deterministic 224×224 ImageNet-normalized tensors derived from official 28×28 source pixels. This avoids a second image archive while preserving provider labels and membership.
2. Provider splits are fixed: train 11,959, validation 1,712, test 3,421. No random re-split occurs.
3. Test data are not loaded by the baseline entry point; validation macro-F1 is the selection metric. Test evaluation needs Project Lead confirmation that validation decisions are locked.
4. The 1% train subset has exactly 120 samples, allocated by proportional largest remainders, seed 2026. Its index file is the reproducibility manifest.
5. Majority baseline predicts the training majority class. Linear probe uses ImageNet-pretrained torchvision ResNet-18 as a frozen 512-D encoder and multinomial sklearn logistic regression fit on training features only.

## Blockers / next action

- **Immediate blocker:** Project Lead must integrate staged implementation and manifest into the tracked repository, then identify a commit hash for Colab.
- **Then:** launch the thin GitHub-imported Colab notebook on a T4, run `python -m bloodmnist_baselines.runner --config configs/baseline_validation.yaml` for 100% then fixed 1%, and record Colab URL, commit hash, dataset checksum/version, configuration, seed, runtime, and validation metrics.
- **Hard constraint:** no baseline training/evaluation is run on the team computer.

## Handoff

Training Engineer uses the official splits and consumes the fixed manifest at `splits/bloodmnist_train_1pct_seed2026.npz`; see `docs/data_and_baselines.md` for counts, class mapping, and exact protocol.
