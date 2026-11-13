# BloodMNIST validation-only ResNet-18 project

## Current status

No experiment result exists yet. The repository contains one simplified, validation-only workflow for three required modes:

- `majority`: predicts the class most frequent in the selected official training data;
- `linear_probe`: ImageNet-pretrained ResNet-18 with only its final layer trainable;
- `finetune`: ImageNet-pretrained ResNet-18 with all parameters trainable.

The test split is locked. The provider constructs only official `train` and `val` datasets; no launcher, configuration, or mode can request test data.

## Layout

```text
config.py            one plain configuration dictionary and mode selection
data.py              one train/validation loader path and safety checks
model.py             majority procedure and pretrained ResNet-18 modes
train.py             direct train and validation loops
run_experiment.py    T4 guard, orchestration, metrics, and evidence saving
src/bloodmnist_baselines/data.py
                      audited official train/validation provider
notebooks/colab_run.ipynb
                      thin Colab launcher only
```

## Required Colab T4 workflow

1. Open `notebooks/colab_run.ipynb` in Google Colab.
2. Select an NVIDIA T4 runtime.
3. The notebook clones this repository, checks out its pinned reviewed commit, installs `requirements-colab.txt`, and invokes `run_experiment.py`.
4. Before running, set only `CONFIG["mode"]` in the pinned `config.py` to `majority`, `linear_probe`, or `finetune`; commit and pin that configuration change first.
5. Download or inspect the created `outputs/<timestamp>_<run-name>_<mode>/` evidence directory.

The notebook contains no data-loading, model, training, or evaluation implementation.

## Safety and data contract

- Official BloodMNIST images are 28×28 and are resized exactly once to 224×224 in `data.py` transforms.
- The provider validates the audited archive identity, official split counts, and class mapping.
- The optional 1% setting requires `splits/bloodmnist_train_1pct_seed2026.npz`; a missing or invalid manifest fails rather than silently using full training data.
- The manifest is applied only to training; validation always uses the full official validation split.
- `best_model.pt` for neural modes is selected only by validation macro-F1.
- No final test evaluation can occur in this workflow because no test dataset is loaded.

## Evidence per Colab run

Every run writes `outputs/.../` with:

- `run_metadata.json`: commit, GPU, package versions, provider/data identity, seed, and test-lock status;
- `config.json`: exact configuration copy;
- `metrics.json`: validation metrics, including macro-F1 and per-class metrics;
- `summary.json`: selected mode, best validation macro-F1, runtime, and explicit `test_evaluated: false`;
- neural modes only: `best_model.pt`, `last_model.pt`, `history.csv`, `learning_curves.png`.

No results are claimed until a Colab T4 run produces these artifacts.
