# BloodMNIST Data, Split, and Mandatory Baseline Protocol

This document is the authoritative interface contract for the data audit and the two required baseline implementations. It accompanies the modular package `src/bloodmnist_baselines/`; workflow logic must not be moved into the Colab notebook.

## Data source, scope, and licensing

- **Dataset:** BloodMNIST (`flag="bloodmnist"`) from [MedMNIST v2](https://medmnist.com/), a 2D, eight-class, RGB blood-cell classification dataset.
- **Source:** MedMNIST project / official Zenodo-hosted NPZ distribution. The 28px archive is named `bloodmnist.npz`; its official MD5 is `7053d0359d879ad8a5505303e11de1dc`. The optional 224px release `bloodmnist_224.npz` has MD5 `b718ff6835fcbdb22ba9eacccd7b2601`.
- **License:** MedMNIST is released under Apache License 2.0; downstream use must also cite MedMNIST and the original BloodMNIST source described by its dataset card. Confirm the current upstream license text when preparing final distribution.
- **Resolution policy:** load the compact official `size=28` data, preserve its official labels/splits, and deterministically resize to 224×224 for ResNet-18. This is semantically equivalent in membership and label data to the official 224px archive while reducing download/storage burden. ImageNet normalization is used because the linear-probe encoder is ImageNet-pretrained.

## Class mapping

| Index | Label |
| ---: | --- |
| 0 | basophil |
| 1 | eosinophil |
| 2 | erythroblast |
| 3 | immature granulocytes (myelocytes, metamyelocytes and promyelocytes) |
| 4 | lymphocyte |
| 5 | monocyte |
| 6 | neutrophil |
| 7 | platelet |

## Fixed provider split audit

The provider split is immutable for this project. Training and validation label totals were audited from the official 28px archive. Test count is recorded but test labels are not used by the validation baseline workflow.

| Class | Train count | Train prevalence | Validation count | Validation prevalence |
| --- | ---: | ---: | ---: | ---: |
| basophil | 852 | 7.12% | 122 | 7.13% |
| eosinophil | 2,181 | 18.24% | 312 | 18.22% |
| erythroblast | 1,085 | 9.07% | 155 | 9.05% |
| immature granulocytes (myelocytes, metamyelocytes and promyelocytes) | 2,026 | 16.94% | 290 | 16.94% |
| lymphocyte | 849 | 7.10% | 122 | 7.13% |
| monocyte | 993 | 8.30% | 143 | 8.35% |
| neutrophil | 2,330 | 19.48% | 333 | 19.45% |
| platelet | 1,643 | 13.74% | 235 | 13.73% |
| **Total** | **11,959** | **100.00%** | **1,712** | **100.00%** |

The locked test split comprises **3,421** images. It is excluded from all hyperparameter selection, architecture selection, and required baseline results until the Project Lead confirms validation-based decisions are locked.

## Reproducible split protocol

1. Call `load_official_splits(..., include_test=False)` to obtain provider `train` and `val` only. It checks expected split sizes.
2. 100% experiments use all 11,959 official training samples.
3. 1% experiments load `splits/bloodmnist_train_1pct_seed2026.npz`, never regenerate a different subset. The manifest holds 120 sorted zero-based positions in the official training split.
4. The manifest was generated with NumPy `default_rng(2026)`, class-stratified sampling without replacement, and largest-remainder allocation of `round(11,959 × 0.01)=120` samples. Regeneration should reproduce identical indices with the provided code.
5. Results must report seed, config, source checksum, commit hash, Colab URL, wall-clock runtime, hardware, and package/runtime versions.

## Modules and public interfaces

| Module | Interface | Inputs → outputs |
| --- | --- | --- |
| `data.py` | `load_official_splits`, `stratified_indices`, `fixed_subset` | Data root / official labels / manifest → validated provider datasets or deterministic subset. |
| `baselines.py` | `majority_class_baseline`, `frozen_resnet18_linear_probe` | Train labels or train+validation datasets → validation probability matrix plus implementation metadata. |
| `metrics.py` | `metric_payload` | Validation labels + eight-column probability scores → accuracy, macro-F1, macro OVR ROC-AUC, per-class precision/recall/F1/support/prevalence. |
| `runner.py` | `run_validation_baselines` / module CLI | YAML configuration → versioned structured JSON evidence. Explicitly rejects a test run. |

## Baseline choices

### Majority-class classifier

The classifier selects `argmax(train_class_count)` and emits a one-hot score vector for that class for every validation case. It has no fitted image parameters. Its ROC-AUC is formally computable from the constant score matrix but should be interpreted cautiously; the complete per-class precision/recall and macro-F1 are retained to show the imbalance-aware floor.

### Frozen ResNet-18 feature linear probe

`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` is truncated before the ImageNet classification head. It is put in evaluation mode and every parameter is frozen. Each image produces a 512-D feature vector. `sklearn.linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial", C=1.0, max_iter=1000, random_state=seed)` fits only linear softmax decision parameters on training features. Validation scores are passed to the shared metric routine.

The current required configuration intentionally fixes `C=1.0`; it is not tuned on test data. Any future validation sweep must be registered and must preserve the test lock.

## Metrics and reporting

All required baseline claims are based on the official validation labels and contain:

- class support and validation prevalence next to per-class precision, recall, and F1;
- validation accuracy and macro-F1 (primary selection measure);
- macro one-vs-rest multiclass ROC-AUC if all classes appear in validation, otherwise `null`;
- configuration, random seed, train/validation cardinalities, device, Python/Torch/CUDA versions, start timestamp, and total runtime.

## Colab execution (after Project Lead integrates a commit)

The sole Colab notebook must remain thin: clone/pull the identified repository commit, install dependencies, set the YAML config, and invoke the module entry point. It must not embed dataset logic, training loops, or metric code.

```bash
pip install -q medmnist scikit-learn pyyaml
export PYTHONPATH="$PWD/src"
python -m bloodmnist_baselines.runner --config configs/baseline_validation.yaml
```

Run the 100% configuration first. Then set `train_subset: 1pct` and repeat using the same fixed manifest. Do not perform final test evaluation from this entry point. Copy generated JSON files from `artifacts/baseline_results/` into version control only after review, and record the Colab link and integrated commit in the status file.
