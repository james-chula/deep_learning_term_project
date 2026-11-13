# Decision log

## 13 November 2026, 11:34 Asia/Bangkok — first fine-tuning simplification

A root-level procedural fine-tuning workflow was introduced to replace a framework-style launcher. It established a thin Colab launcher, validation macro-F1 checkpoint selection, provenance capture, and a locked test policy.

## 13 November 2026, 13:12 Asia/Bangkok — simplify all workflows

Authorized scope expanded to majority-class baseline, frozen ResNet-18 linear-probe baseline, and fine-tuning. The required outcome is one small, modular `.py` workflow, thin GitHub-imported Colab launcher, no test usage, and deletion of obsolete code only after replacement verification.

## 13 November 2026, 13:19 Asia/Bangkok — accept and push consolidated design

Accepted design: one `CONFIG["mode"]` selects `majority`, `linear_probe`, or `finetune`; all modes use one audited official train/validation provider. The 1% subset remains `splits/bloodmnist_train_1pct_seed2026.npz` with seed 2026 and applies only to training. Official 28×28 images resize once to 224×224. Validation macro-F1 selects neural checkpoints. The test split has no active loading path and all run evidence records `test_evaluated: false`.

Removed/superseded files are recorded in `PROJECT_STATE.md` with the consolidation commit once pushed. No experiment result is implied by implementation or static verification.
