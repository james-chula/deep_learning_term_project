# Fine-tuned ResNet-18 Training Pipeline — Integration Specification

**Owner:** Training Engineer
**Prepared:** 2026-11-13 10:29 Asia/Bangkok
**Prepared against documentation reference:** `fded1f25216f00a0cdf09a46be6fe39c8c19dab8`; implementation must use the Project Lead-designated post-integration commit.

## 1. Scope and safeguards

This design implements the first training condition only: ImageNet-pretrained ResNet-18 fine-tuned for MedMNIST `bloodmnist` at image size 224. The provider train/validation split is consumed exactly as defined by the manifest:

- train: 11,959; validation: 1,712; locked test: 3,421
- labels: 0 basophil, 1 eosinophil, 2 erythroblast, 3 immature granulocytes, 4 lymphocyte, 5 monocyte, 6 neutrophil, 7 platelet
- expected 224 archive MD5: `b718ff6835fcbdb22ba9eacccd7b2601`

The standard `train` entry point must reject `test` as an evaluation split. A separately named `evaluate_locked_test` command is intentionally **not** included until the Project Lead gives explicit post-lock authorization. All tuning decisions use validation metrics only.

No training is run outside a Project-Lead-confirmed Google Colab T4 session after the selected source commit has been pulled.

## 2. Proposed module layout

```text
configs/
  bloodmnist_resnet18_finetune.yaml       # canonical first-run config
src/bloodcell/
  __init__.py
  cli.py                                  # argparse command dispatcher; python -m bloodcell.cli train --config ...
  orchestrate.py                          # single clear programmatic entry point: run_training(config_path, overrides)
  config.py                               # YAML load, schema/dataclass validation, resolved-config serialization
  reproducibility.py                      # seed setup, determinism policy, environment/GPU capture
  data.py                                 # manifest-driven BloodMNIST dataset construction and DataLoaders
  transforms.py                           # named train/validation transform builders
  models.py                               # ResNet-18 construction; pretrained weights and 8-logit head replacement
  engine.py                               # one-epoch train/validate loops, AMP, checkpoint decisions
  metrics.py                              # loss, accuracy, per-class PRF, macro-F1, multiclass ROC-AUC when valid
  checkpoints.py                          # atomic save/load, best/last checkpoint, resume compatibility checks
  artifacts.py                            # run directory, JSON/CSV ledgers, curves, reports
  reporting.py                            # JSON-safe run summary and human-readable validation report
  schemas.py                              # typed config/result dataclasses and validation helpers
notebooks/
  colab_train_resnet18.ipynb              # THIN launcher only; no workflow logic
scripts/
  train.py                                # optional compatibility wrapper calling bloodcell.cli.main()
docs/state/
  training_status.md                      # persistent state, updated by owner
```

Imports should have no GPU side effects. Dataset download/verification, model creation, and run-directory creation occur only when `run_training()` is invoked.

## 3. Clear entry point and CLI

Canonical invocation:

```bash
python -m bloodcell.cli train \
  --config configs/bloodmnist_resnet18_finetune.yaml \
  --run-root /content/drive/MyDrive/bloodcell_runs \
  --source-commit <FULL_GIT_SHA>
```

Optional documented CLI flags:

- `--config PATH` (required): YAML experiment config.
- `--run-root PATH` (optional): override config artifact root; must be writable persistent storage in Colab (prefer mounted Drive).
- `--source-commit SHA` (required for evidence runs): exact checked-out SHA recorded verbatim; validation requires a 40-hex SHA.
- `--resume PATH` (optional): `last.pt` checkpoint only; must pass compatibility checks.
- `--set dotted.path=value` (repeatable): narrow, logged config overrides for scratch work. Every override is saved in the resolved config and ledger.
- `--dry-run` (optional): validate config, manifest, split counts, and model shape; no epochs and no metrics claim.

The orchestrator is `bloodcell.orchestrate.run_training(config_path, run_root=None, source_commit=None, resume_path=None, overrides=None) -> RunResult`. The CLI only parses arguments, invokes it, prints the final artifact directory, and exits nonzero on a failed/incomplete run.

Thin Colab notebook cells are limited to: (1) verify T4 availability, (2) clone/pull the Project-Lead-designated SHA, (3) install pinned requirements, (4) mount persistent artifact storage if selected, (5) set paths/commit variables, and (6) execute the canonical command. It must not implement datasets, transforms, loops, metrics, checkpointing, or plotting.

## 4. Config schema

A strict YAML schema is recommended; unknown keys fail validation. Example canonical config:

```yaml
experiment:
  name: bloodmnist_resnet18_finetune
  purpose: validation_only_model_selection
  seed: 20261113
  output_root: /content/drive/MyDrive/bloodcell_runs

data:
  dataset_flag: bloodmnist
  archive_size: 224
  archive_md5: b718ff6835fcbdb22ba9eacccd7b2601
  manifest_path: configs/data/bloodmnist_224_fixed_split.yaml
  num_classes: 8
  class_names: [basophil, eosinophil, erythroblast, immature_granulocytes, lymphocyte, monocyte, neutrophil, platelet]
  expected_split_counts: {train: 11959, val: 1712, test: 3421}
  train_split: train
  validation_split: val
  test_policy: locked_not_loaded
  num_workers: 2
  pin_memory: true
  persistent_workers: false

model:
  architecture: resnet18
  initialization: imagenet1k_v1_pretrained
  pretrained_weights: ResNet18_Weights.IMAGENET1K_V1
  classifier_out_features: 8
  trainable_policy: all_layers

transforms:
  train:
    resize: 224
    random_horizontal_flip_p: 0.5
    random_rotation_degrees: 10
    normalize_mean: [0.485, 0.456, 0.406]
    normalize_std: [0.229, 0.224, 0.225]
  validation:
    resize: 224
    normalize_mean: [0.485, 0.456, 0.406]
    normalize_std: [0.229, 0.224, 0.225]

optimization:
  epochs: 20
  batch_size: 64
  loss: cross_entropy
  optimizer: adamw
  lr: 0.0001
  weight_decay: 0.0001
  scheduler: cosine_annealing_lr
  scheduler_t_max: 20
  amp: true
  gradient_clip_norm: null

selection:
  monitor: val_macro_f1
  mode: max
  save_best_only: true
  report_roc_auc: true

runtime:
  device: cuda_required
  gpu_requirement: T4
  deterministic_algorithms: false
  cudnn_benchmark: true
```

The data manifest referenced above remains Data & Baseline Engineer-owned. It must declare official source URL/identity, split policy, archive checksum, label mapping, and expected counts. Training must stop if its declared values conflict with the fixed handoff.

**Modeling rationale:** pretrained ImageNet ResNet-18 is the stipulated initial condition; replacement of only the final fully connected layer produces eight logits. ImageNet normalization matches pretrained weight expectations. Mild horizontal flip and rotation apply to training only; validation is deterministic resize + normalization. AdamW with a conservative fine-tuning learning rate is a transparent initial condition, not a tuned claim.

## 5. Artifact, result, and ledger schemas

Run directory (generated as `<UTC-or-Bangkok timestamp>_<experiment>_<sha8>_seed<seed>`; store timestamps with timezone):

```text
<run_dir>/
  resolved_config.yaml
  run_metadata.json
  epoch_metrics.csv
  best.pt
  last.pt
  best_validation_metrics.json
  classification_report_val.json
  confusion_matrix_val.csv
  curves.png
  run_summary.json
  stdout.log
```

`run_metadata.json` records this minimum evidence contract:

```json
{
  "run_id": "...",
  "status": "running|completed|failed|resumed",
  "source_commit": "40-char SHA",
  "started_at": "ISO-8601 with +07:00",
  "finished_at": "ISO-8601 with +07:00 or null",
  "wall_clock_seconds": null,
  "host": {"runtime": "Google Colab", "python": "...", "torch": "...", "torchvision": "..."},
  "gpu": {"required": "T4", "name": "NVIDIA Tesla T4", "cuda": "...", "device_count": 1},
  "seed": 20261113,
  "dataset": {"flag": "bloodmnist", "size": 224, "archive_md5": "...", "manifest_path": "...", "split_counts": {"train":11959,"val":1712,"test":3421}, "test_policy":"locked_not_loaded"},
  "model": {"architecture":"resnet18", "initialization":"imagenet1k_v1_pretrained", "trainable_policy":"all_layers"},
  "transforms": {"train":"serialized config", "validation":"serialized config"},
  "optimization": {"optimizer":"AdamW", "lr":0.0001, "scheduler":"CosineAnnealingLR", "epochs_requested":20, "batch_size":64},
  "artifacts": {"best_checkpoint":"best.pt", "last_checkpoint":"last.pt", "epoch_ledger":"epoch_metrics.csv"}
}
```

`epoch_metrics.csv` is append-only and includes: `epoch, phase, loss, accuracy, macro_f1, learning_rate, samples, elapsed_seconds`. One `train` and one `val` row per completed epoch. `best_validation_metrics.json` includes the selected epoch and validation accuracy, macro-F1, per-class precision/recall/F1/support, confusion matrix reference, and multiclass one-vs-rest ROC-AUC macro plus per-class values when all classes/probabilities permit calculation. The final summary must identify the **best validation checkpoint**, never test metrics.

At run start and termination, append a one-record JSON object to `<run_root>/run_ledger.jsonl`. It is a cross-run index containing `run_id`, source SHA, config hash/path, seed, initialization, status, time, best validation epoch/macro-F1 (only at termination), runtime, GPU name, artifact directory, and failure reason if any. Do not overwrite historical records.

## 6. Checkpoint and resume contract

- Save `last.pt` at every fully completed epoch and `best.pt` whenever `val_macro_f1` strictly improves. Tie behavior: retain the earlier checkpoint for deterministic selection.
- Checkpoints include: epoch, model state, optimizer state, scheduler state, AMP scaler state, best monitor value/epoch, RNG states (Python/NumPy/Torch CPU/CUDA), resolved-config hash, source commit, and partial metric history.
- Write checkpoint to a temporary file then atomically rename to avoid corrupting the prior checkpoint.
- Resume is allowed **only** from `last.pt`; `best.pt` is inference/selection only. On resume, reject a mismatch in architecture, class mapping/count, archive size/checksum, manifest identifier, split counts, transform config, optimizer/scheduler, batch size, seed, initialization, or source commit unless an explicit future migration policy is approved. This prevents accidental comparison of incompatible runs.
- A resumed run retains its original `run_id`, records every resume event in `run_metadata.json`, and writes `resumed` status before continuation. It resumes at `checkpoint_epoch + 1` and does not duplicate a completed epoch ledger row.
- If the runtime is not CUDA or the detected GPU name does not contain `T4`, exit before training with a clear error and write a failed metadata record. This is an enforcement aid; the user still verifies Colab hardware.

## 7. Required validation reporting flow

1. Validate config and Data Engineer manifest; verify label mapping, expected 224 checksum, and provider split counts.
2. Establish seed and record reproducibility/runtime environment.
3. Require CUDA T4, construct fixed datasets/loaders and pretrained ResNet-18.
4. Train/validate each epoch, saving complete ledger rows and curves data.
5. Select `best.pt` by validation macro-F1 only.
6. Re-load `best.pt` and calculate saved validation-only final report: loss, accuracy, per-class precision/recall/F1/support, macro-F1, ROC-AUC where defined, confusion matrix.
7. Save plots for train/validation loss and macro-F1; emit evidence paths and summary.
8. Do not load or score the locked test split.

## 8. Integration deliverables / changed paths

For the first integrated implementation, create the paths listed in Section 2 plus package metadata and pinned requirements as used by the repository. Ensure each module has a module docstring, typed public API, and concise non-obvious implementation comments. Add unit tests that do not train: config schema rejection, split-policy rejection, ResNet head shape, metric serialization, checkpoint compatibility/atomic writer, and resume epoch behavior.

Update `docs/state/training_status.md` when this design is integrated, when a run starts/ends, and whenever the current commit, config, artifacts, blocker, or next action changes. Each update includes Asia/Bangkok time, branch/commit (when supplied by Project Lead), validation-only restriction, artifact links, and handoff status.
