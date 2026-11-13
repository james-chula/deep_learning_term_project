# Decision log

## 13 November 2026, 11:34 Asia/Bangkok — first fine-tuning simplification

A root-level procedural fine-tuning workflow was introduced to replace a framework-style launcher. It established a thin Colab launcher, validation macro-F1 checkpoint selection, provenance capture, and a locked test policy.

## 13 November 2026, 13:12 Asia/Bangkok — simplify all workflows and remove unnecessary files

Authorized scope expanded to majority-class baseline, frozen ResNet-18 linear-probe baseline, and fine-tuning. The required outcome is one small modular `.py` workflow, a thin GitHub-imported Colab launcher, no test usage, and removal of obsolete code only after replacement verification.

Retention requirements: keep project state, this decision log, Colab reproducibility instructions, fixed data/subset audit records, baseline and fine-tuning descriptions, and validation-only/locked-test safeguards. A deletion is acceptable only when the item is unreferenced by the active launcher/config/modules and is not the sole provenance, reproducibility, artifact-interpretation, or safeguard record.

## 13 November 2026, 13:19 Asia/Bangkok — accept consolidated design pending final pin confirmation

Accepted design: one `CONFIG["mode"]` selects `majority`, `linear_probe`, or `finetune`; `CONFIG["use_fixed_1pct_subset"]` selects the required training-data condition. `True` uses the versioned fixed 1% membership (`splits/bloodmnist_train_1pct_seed2026.npz`, 120 examples, seed 2026); `False` uses the complete official training split. Both conditions use the same full official validation split and validation macro-F1 for neural checkpoint selection.

All modes use one audited official train/validation provider. The active execution input is the official 28×28 archive, then a single deterministic resize to 224×224; any separately distributed 224px archive checksum is reference-only. The test split has no active loading path and all run evidence records `test_evaluated: false`. No experiment result is implied by implementation or static verification.

## 13 November 2026, 14:00 Asia/Bangkok — documentation reconciliation after simplification

README and project state were reconciled to document the 1% versus 100% comparison, active archive/checksum policy, thin-notebook boundary, evidence schema, and result-reporting requirements. At this time no Colab T4 experiment evidence exists. The Project Lead must confirm the final pushed source commit and exact Colab pin before any execution; prior stable-base/provider/launcher references remain provenance, not authorization to run a later consolidated workflow.

## 13 November 2026, 14:10 Asia/Bangkok - provider returns ProviderDatasets

The shared provider build_provider_datasets(*, audit_manifest_path, image_size, train_transform, val_transform, subset_manifest_path=None) now returns the frozen dataclass ProviderDatasets(train, val, metadata), exported from bloodmnist_baselines, matching the canonical contract Project Lead reported for SHA 8a637d34017bce6311225b82fd5d3a6563d7dd26. Root data.py uses attribute access. Subset validation is one private train-only helper; no test constructor or option exists. Static checks only; no experiment and no Git action.

## 13 November 2026, 14:11 Asia/Bangkok — defer controlled-matrix interface

User approved deferral of the controlled-matrix interface: augmentation on/off, learning-rate candidate/stage, seed-role, and expanded comparable-condition metadata are not included in this stabilization push. The existing committed configuration still documents the 1% versus 100% data-fraction selection. No controlled-matrix run is authorized until the deferred controls and evidence contract are implemented and reviewed.


## 13 November 2026, 14:40 Asia/Bangkok — narrowly normalize provider class-label whitespace

The audit/provider class mapping comparison may ignore repeated whitespace and whitespace immediately before an opening parenthesis. Class IDs and all semantic label text remain strict. This addresses the MedMNIST 3.0.2 class-3 formatting difference without changing the audited artifact or weakening provenance validation.
