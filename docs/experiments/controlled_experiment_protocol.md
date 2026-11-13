# Controlled Experiment Protocol and Decision Log

**Status:** Approved by Project Lead; pending documentation commit
**Prepared:** 2026-11-13 10:24 Asia/Bangkok
**Applies to:** BloodMNIST / ResNet-18 term project, post-progress-report experiments

## 1. Objective

Measure how transfer-learning treatment and training-data availability affect validation performance while preventing leakage from the held-out test split.

**Primary selection metric:** validation macro-F1.
**Secondary validation metrics:** macro one-vs-rest ROC-AUC, accuracy, per-class precision/recall, confusion matrix, and runtime. Report class prevalence alongside metric tables.
**No test-set selection:** do not load test labels or calculate test metrics during tuning, strategy selection, or augmentation selection.

## 2. Non-negotiable comparison controls

For a given data-size cell, each strategy uses:

1. the same official validation split;
2. the same fixed training membership (the committed stratified 1% subset or all official training examples);
3. identical image resize/crop policy, ImageNet normalization, batch size, epoch budget, optimizer family, scheduler policy, loss, label mapping, mixed-precision policy, dataloader settings, and validation evaluator;
4. no training augmentation in the main comparison;
5. identical checkpoint-selection rule: select the epoch with the highest validation macro-F1, with deterministic tie handling documented in config;
6. the same three fixed run seeds (`42`, `123`, `2026`) for confirmation runs; and
7. an equal number of candidate learning rates per strategy.

A training budget means **maximum optimizer epochs over the same selected data-size cell**, not a matched number of update steps. Log both epochs completed and optimizer steps so any early termination is visible. LR-selection sweeps use the same maximum budget as confirmation runs unless the decision log records and justifies a cheaper, uniformly applied screening budget.

## 3. Strategies

| ID | Strategy | Initialization / trainable parameters |
| --- | --- | --- |
| `linear_probe_imagenet` | Frozen linear probe | torchvision ImageNet-pretrained ResNet-18 backbone frozen; replace classifier with trainable task head. |
| `finetune_imagenet` | Full fine-tuning | Same ImageNet-pretrained ResNet-18; all backbone and head parameters trainable. |
| `random_init` | From-scratch training | ResNet-18 created without pretrained weights; all parameters trainable. |

All arms must use the same ResNet-18 variant and identical eight-class classifier-head shape. Pretrained-weight identifier, torchvision/PyTorch versions, and trainable-parameter count must be logged.

## 4. Matrix and run accounting

### 4.1 Main comparison — no augmentation

| Data size | Strategy count | LR candidates per strategy | LR-selection seed | Confirmation seeds | Planned runs |
| --- | ---: | ---: | ---: | ---| ---: |
| 1% fixed stratified subset | 3 | 3 | 42 | 42, 123, 2026 | 18 |
| 100% official training split | 3 | 3 | 42 | 42, 123, 2026 | 18 |

**Planned main evidence records: 36** (18 LR-selection records + 18 confirmation records). This is at most 36 physical executions; it is 30 if the seed-42 LR-selection execution is exactly reused as the seed-42 confirmation evidence. A failed/invalid run is not evidence and must be rerun with the same assigned cell only after its cause is logged.

**Default LR candidate rule (requires config confirmation):** use three log-spaced rates within a strategy-specific, documented range. The numeric ranges may differ by strategy because their trainable parameter sets differ, but the **number of candidates, seed, epoch budget, and selection metric** must be equal. Record the exact candidate list before execution; do not add candidates after observing outcomes without opening a new documented decision.

**LR selection:** within each `(data_size, strategy)`, select the candidate with highest validation macro-F1 from seed 42. Resolve exact ties by lower validation loss, then lower LR. The selected candidate is fixed before all three confirmation seeds execute. Its seed-42 result may be reused in the three-seed aggregate only if the exact config and run evidence are identical; otherwise run a fresh seed-42 confirmation and record why.

### 4.2 Augmentation ablation — validation gated

After finishing the main comparison, select **one** strategy and data-size condition according to the predeclared rule below. Hold its selected LR and all other controls fixed.

| Condition | Seeds | Training transform |
| --- | ---| --- |
| `no_aug` | 42, 123, 2026 | Deterministic preprocessing only |
| `train_aug` | 42, 123, 2026 | Training-only horizontal flip, vertical flip, and random rotation up to 15°; validation remains deterministic |

**Total planned augmentation runs: 6.** Do not introduce validation/test augmentation. Exact probabilities, interpolation/fill settings, and transform order must be captured in config.

## 5. Validation-only decision gates

1. **LR gate:** select LR separately for each strategy × data-size cell on validation macro-F1 only.
2. **Main-strategy gate:** compute seed mean and sample SD of validation macro-F1 for every completed main cell. Select the highest mean among the predeclared comparison cells. If means are tied at the reported precision, use lower SD, then higher mean validation macro ROC-AUC; record this tie-break in the decision log.
3. **Augmentation gate:** compare validation macro-F1 mean ± sample SD for augmentation and no augmentation within the selected strategy/data-size cell. Adopt augmentation only if it improves the validation-only selection rule stated before viewing its results. Recommended rule: higher mean macro-F1; ties use lower SD then macro ROC-AUC.
4. **Lock gate:** Project Lead explicitly confirms that strategy, data size, augmentation, LR/hyperparameters, preprocessing, seed protocol, and checkpoint rule are locked.
5. **Test gate:** only after the lock gate, evaluate the final locked procedure on the official test split once. Report that result as confirmatory, not as a tuning signal.

## 6. Run ledger schema

Store one machine-readable row per attempted run (CSV/JSONL) and one human-readable summary table. Required fields:

| Field | Requirement |
| --- | --- |
| Run identity | `run_id`, timestamp with timezone, commit hash, Colab notebook/run URL, command/entry point |
| Data identity | dataset source/version/checksum where available, split file path/checksum, data size, per-class counts |
| Configuration | strategy, pretrained-weight identifier, seed, exact LR/candidate list, batch size, epoch budget, optimizer/scheduler, transforms, precision/device |
| Resources | Colab GPU model, runtime duration, epochs/steps completed, peak GPU memory if available |
| Validation evidence | selected epoch, loss, accuracy, macro-F1, macro ROC-AUC, per-class precision/recall, confusion-matrix path, curve paths, checkpoint path |
| Integrity | status (`complete`, `invalid`, `failed`, `interrupted`), exception/reason, resume provenance, output-artifact paths |
| Analysis | inclusion flag, exclusion rationale, decision affected, analyst review date |

A run is **invalid/incomparable** if it has different split membership, wrong/unknown commit, test-informed tuning, differing uncontrolled preprocessing or epoch budget, missing seed/config evidence, mismatched strategy definition, incomplete validation metrics, or an unlogged resume/checkpoint. Retain it in the ledger but exclude it from aggregates, with a specific reason.

## 7. Aggregation and tables

For every completed three-seed confirmation cell, report `mean ± sample SD` (`ddof=1`) across valid runs. Do not substitute standard error for SD. If fewer than three valid seeds exist, label the result provisional and show every seed; do not make a final selection from it unless the Project Lead records an exception.

### Main-table specification

| Data size | Strategy | Selected LR | Seeds valid/attempted | Val macro-F1 (mean ± SD) | Val macro ROC-AUC (mean ± SD) | Val accuracy (mean ± SD) | Runtime (mean ± SD) | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |

Attach a supplementary per-seed table, per-class precision/recall table with prevalence, and a validity/exclusions table. Use a grouped bar chart of validation macro-F1 (x: strategy; facets: 1%/100%; error bars: sample SD; include individual seed points). Use a second grouped bar chart for macro ROC-AUC. Do not graph test metrics during selection.

### Augmentation-table specification

| Selected strategy | Data size | Train transform | Seeds valid/attempted | Val macro-F1 (mean ± SD) | Val macro ROC-AUC (mean ± SD) | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |

## 8. Required software architecture and documentation

- Put training, data transforms, models, evaluation, configuration parsing, logging, and aggregation in modular `.py` modules; avoid substantive notebook workflow logic.
- Provide a single documented orchestration/entry-point command that runs one declared configuration.
- Keep one thin GitHub-imported Colab notebook: environment/setup, repository load/update to a Lead-designated commit, config selection, and entry-point invocation only.
- Record interface changes and all experimental decisions in this document or a linked decision log. Save configs and run outputs as versioned artifacts where practical.
- The Project Lead alone performs Git operations and selects the commit to pull in Colab. All GPU experiments run only on a Colab T4 after that confirmation.

## 9. Decision log

| Date (Asia/Bangkok) | Decision | Rationale / controls | Evidence | Owner / approval |
| --- | --- | --- | --- | --- |
| 2026-11-13 | Drafted 3 × 2 no-augmentation matrix with equal-size (3-LR) sweeps and seeds 42/123/2026. | Satisfies controlled strategy/data-size comparison and multi-seed aggregation. | This protocol; `progress_tracker.md`. | Experiment Coordinator draft; reviewed and approved by Project Lead; pending commit. |
| 2026-11-13 | Final test evaluation remains locked. | Preserve test set as a one-time confirmatory evaluation after validation-only choices are frozen. | Project Lead coordination message. | Project Lead authority. |
