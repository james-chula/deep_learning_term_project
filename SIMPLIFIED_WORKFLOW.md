# Simplified validation-only workflow

## Approved design

The project uses five procedural root modules and one locked provider module:

1. `config.py`: plain `CONFIG` dictionary; selects `majority`, `linear_probe`, or `finetune`.
2. `data.py`: creates train/validation transforms and loaders through the provider.
3. `model.py`: majority predictions and pretrained eight-class ResNet-18 creation.
4. `train.py`: direct training and validation routines.
5. `run_experiment.py`: T4 check, seed, orchestration, validation-only selection, and evidence serialization.
6. `src/bloodmnist_baselines/data.py`: validated official train/validation source with no test-loading API.

The mandatory execution sequence is:

```text
T4 check -> seed -> train/validation data -> selected mode -> validation -> save evidence
```

## Modes

| Mode | Training rule | Checkpoint selection |
|---|---|---|
| `majority` | No neural model; use selected training labels only | Not applicable |
| `linear_probe` | Freeze pretrained ResNet-18 backbone; update final layer only | Validation macro-F1 |
| `finetune` | Update all pretrained ResNet-18 parameters | Validation macro-F1 |

## Safety invariants

- Provider constructs only official training and validation datasets.
- Test data cannot be requested through the active API.
- Source image size is checked as 28 and transformed once to 224.
- A requested 1% run requires the versioned fixed-subset manifest and cannot fall back to full training data.
- Validation does not call an optimizer or update weights.
- Neural `best_model.pt` is saved only when validation macro-F1 improves.
- All summaries record `test_evaluated: false`.

## Thin Colab notebook

`notebooks/colab_run.ipynb` may only: verify T4, clone the repository, check out its pin, install requirements, invoke `run_experiment.py`, and list saved evidence. It contains no workflow implementation.
