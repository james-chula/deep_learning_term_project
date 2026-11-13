# BloodMNIST validation-only ResNet-18 project

## Current status

No experiment result exists yet. This repository contains one simplified, validation-only workflow for three conditions:

- `majority`: predicts the class most frequent in the selected official training data;
- `linear_probe`: ImageNet-pretrained ResNet-18 with only its final layer trainable;
- `finetune`: ImageNet-pretrained ResNet-18 with all parameters trainable.

The test split is locked. The active provider constructs only official `train` and `val` datasets; no launcher, configuration, or mode can request test data. Implementation and documentation are not experimental evidence: no validation metric, runtime, checkpoint, curve, or result artifact is reportable until a Colab T4 run saves its evidence directory.

## Layout and public interfaces

```text
config.py            one plain configuration dictionary and mode/data-fraction selection
data.py              train/validation transforms, loaders, and safety checks
model.py             majority procedure and pretrained ResNet-18 modes
train.py             direct train and validation loops
run_experiment.py    T4 guard, orchestration, metrics, and evidence saving
src/bloodmnist_baselines/data.py
                      audited official train/validation provider; no test API
notebooks/colab_run.ipynb
                      thin Colab launcher only
artifacts/data_audit/audit_manifest.json
                      dataset/split/subset audit contract
splits/bloodmnist_train_1pct_seed2026.npz
                      fixed 1% training membership
```

`run_experiment.py` is the single execution entry point. `CONFIG` in `config.py` is its public configuration interface: select one `mode`, give a run name and seed, and set `use_fixed_1pct_subset` to select the fixed 1% training condition (`True`) or the full official-training condition (`False`). The selected data fraction changes **training data only**; validation always uses the complete official validation split.

## Reproducible 1% versus 100% plan

The required comparison uses the same official validation split, metric, seed policy, preprocessing, and mode definition at two training fractions:

| Training condition | Configuration | Training membership | Validation / selection |
|---|---|---|---|
| 1% | `use_fixed_1pct_subset=True` | Versioned fixed manifest; 120 examples; seed 2026 | Full official validation set; macro-F1 for neural checkpoint selection |
| 100% | `use_fixed_1pct_subset=False` | All 11,959 official training examples | Same full official validation set; macro-F1 for neural checkpoint selection |

Do not replace the fixed 1% membership with a newly sampled subset. A planned condition is not complete until its Colab evidence is saved. Any later multi-seed aggregation or final test evaluation requires an explicitly approved protocol; final test evaluation is outside this workflow.

## Required Colab T4 workflow

1. Confirm the Project Lead's designated, pushed commit and use that exact reviewed pin. Do not substitute an earlier pin.
2. Open `notebooks/colab_run.ipynb` in Google Colab and select an NVIDIA T4 runtime.
3. The notebook verifies T4, clones the repository, checks out its pin, installs `requirements-colab.txt`, and invokes `PYTHONPATH=src python run_experiment.py`.
4. Before a run, select the intended `CONFIG["mode"]` and training fraction in `config.py`; commit and pin that configuration change before Colab execution.
5. Inspect or download the resulting `outputs/<timestamp>_<run-name>_<mode>/` evidence directory. Retain it with its source commit and configuration.

The notebook contains no data-loading, model, training, or evaluation implementation. It is only setup, version selection, dependency installation, entry-point invocation, and evidence listing. The workflow requires a T4 and rejects CPU or non-T4 execution.

## Safety and data contract

- Execution input is the official 28×28 BloodMNIST archive. Its required MD5 is `7053d0359d879ad8a5505303e11de1dc`; the audited local SHA-256 is recorded in `artifacts/data_audit/audit_manifest.json`.
- Official provider split sizes are 11,959 train / 1,712 validation / 3,421 locked test. The test count is provenance only: test data are never loaded by the active workflow.
- Official 28×28 images are resized exactly once to 224×224 in `data.py` transforms. A separately distributed 224px archive checksum is reference-only, not the active execution-input checksum.
- The fixed 1% setting requires `splits/bloodmnist_train_1pct_seed2026.npz`; a missing or invalid manifest fails rather than silently using full training data.
- The fixed manifest applies only to training; validation always uses the full official validation split.
- `best_model.pt` for neural modes is selected only by validation macro-F1. Validation does not update weights.
- No final test evaluation can occur in this workflow because no test dataset is loaded. Do not use validation results to claim final test performance.

## Evidence per Colab run

Every run writes `outputs/.../` with:

- `run_metadata.json`: source commit, GPU, package versions, provider/data identity, seed, subset identity, and test-lock status;
- `config.json`: exact configuration copy;
- `metrics.json`: validation accuracy, macro-F1, valid multiclass OVR ROC-AUC where defined, and per-class precision/recall/F1/support;
- `summary.json`: selected mode, best validation macro-F1, runtime, and explicit `test_evaluated: false`;
- neural modes only: `best_model.pt`, `last_model.pt`, `history.csv`, and `learning_curves.png`.

A result is reportable only when these files are traceable to the reviewed commit, the saved configuration, the T4 run, and the audited data/subset contract.
