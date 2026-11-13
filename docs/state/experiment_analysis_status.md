# Experiment analysis status

**Last updated:** 13 November 2026, 14:08 Asia/Bangkok
**Branch / commit context:** Project Lead reports current shared-checkout HEAD `4795c98` with uncommitted reviewed changes. This is pre-commit review context only; the final Colab pin is pending a pushed, Lead-selected commit. No Git action performed by Experiment Coordinator.
**Evidence scope:** static repository review only; no experiment was run and no validation metric is reported here.

## Governing experiment contract

- Compare `majority`, ImageNet-pretrained frozen ResNet-18 `linear_probe`, and ImageNet-pretrained ResNet-18 `finetune`.
- Cover fixed 1% and 100% training-data conditions, equal training budgets within each comparison, fixed seed records, learning-rate selection, and a gated augmentation ablation.
- The fixed 1% training manifest is `splits/bloodmnist_train_1pct_seed2026.npz`: 120 samples, seed 2026, SHA-256 `11d34a47d4a048a954b3ade270e5abda003f181237b24d34e07fe974ef9c035d` (Lead-provided data contract).
- Select configurations and checkpoints from validation only. Do **not** evaluate the final test split until Project Lead confirms all choices are locked.

## Current repository-review finding

The simplified root-module design preserves key safeguards: a thin Colab notebook, a single `run_experiment.py` entry point, explicit T4 guard, and no active provider API for test data. It is a plausible execution base, but is **not yet approved as experiment-ready** for the controlled matrix.

Remaining blocking issues for valid controlled evidence:

1. Current `CONFIG` exposes one fixed configuration (currently 1%) and has no documented matrix/config interface for 100%, LR candidates, confirmation seeds, or augmentation on/off. Manual edits alone do not provide a durable comparable-run contract.
2. `RandomHorizontalFlip()` is hard-coded for every training run, so the required augmentation ablation has no no-augmentation control.
3. Explicit run provenance should include data-fraction/condition, augmentation condition, LR candidate/selection stage, and exact subset identity in the compact experiment ledger. Existing run metadata is a useful base but does not yet enforce the controlled-matrix fields.

**Resolved in the current working tree; pending Project Lead verification and commit:** the prior best-versus-last metrics mismatch. The reviewed working-tree change is reported to align persisted metrics with the validation-macro-F1-selected checkpoint. It is not accepted as committed experiment evidence until Lead verification and a pushed Colab pin.

## Execution and evidence state

| Item | Status | Evidence / path |
|---|---|---|
| Data audit / fixed 1% manifest | Contract reported integrated | Lead-reported commit `5db586c7b37b3585051c88379502ab6c9f2e546c`; manifest path above |
| Simplified implementation | Static review only | `SIMPLIFIED_WORKFLOW.md`, `run_experiment.py`, `config.py` |
| Colab T4 run | Not authorized / not evidenced | No output directory or Colab link supplied |
| Validation results | None | No evidence records |
| Final test evaluation | Locked | Validation-only policy |

## Required next action

**Owner:** Training Engineer / Project Lead.
**Request:** verify and commit the working-tree best-metrics correction; resolve the three remaining experiment-readiness issues above; document the config/result schema; then have Project Lead select and pin a pushed reviewed commit for Colab. The Experiment Coordinator will then register validation-only evidence and aggregate only comparable runs.

## Decision log

- **13 November 2026, 14:00 Asia/Bangkok:** Restored this owned persistent analysis-state file after simplification removed the prior state/tracker documentation. The restoration is documentation-only and creates no experiment claim. The matrix protocol remains to be re-integrated or linked before authorization.
- **13 November 2026, 14:08 Asia/Bangkok:** Updated pre-commit context per Project Lead: HEAD `4795c98`, uncommitted reviewed changes, and no final Colab pin until a pushed Lead-selected commit exists. Reclassified the best-versus-last metrics issue as resolved in the working tree but pending Lead verification/commit; no experiments were run.
