# Documentation: BloodMNIST Data Audit and Fixed 1% Subset

This document explains the purpose and execution of [`01_data_audit_and_subset.ipynb`](01_data_audit_and_subset.ipynb). The notebook is the first practical step of the Blood Cell Classification project.

## Purpose

The notebook prepares the dataset foundation required before model training. It:

1. Downloads and loads the official BloodMNIST splits.
2. Verifies the image format and class labels.
3. Calculates training and validation class counts and prevalence.
4. Displays labelled examples from every class.
5. Creates a fixed, class-stratified 1% training subset.
6. Saves reproducibility artifacts for later experiments and reporting.

It is designed to satisfy the Track 1 data-description requirements and the proposal-feedback requirements.

## Why this notebook is needed

The Track 1 instructions require a data table with the training, validation, and test splits, including class prevalence. They also require that data splits are established before model training.

The proposal feedback additionally requires a meaningful limited-data experiment:

- compare models using **1%** and **100%** of the official training data;
- use a fixed, class-stratified small subset;
- save that subset in the repository.

The notebook creates the fixed 1% subset used by every limited-data experiment. It is approximately 120 images because 1% of the 11,959-image training split is 119.59, rounded to 120.

## Before running

Start from the repository root:

```bash
nix develop
./scripts/setup-runtime
./scripts/jupyter
```

Open `01_data_audit_and_subset.ipynb` in JupyterLab and select the **Python (BloodMNIST ResNet)** kernel.

`setup-runtime` installs the Python package `medmnist` into the local `.venv` when necessary. It does **not** download BloodMNIST data. The dataset download begins only when the notebook's loading cell runs.

## Notebook walkthrough

### 1. Import packages and locate the project root

The first code cell imports the libraries used by the notebook:

- `Path` for file paths;
- `Counter` and `csv` for count tables;
- `json` for audit metadata;
- `numpy` for labels, random sampling, and saving indices;
- `pandas` for readable tables;
- `matplotlib` for displaying labelled images;
- `medmnist` for the BloodMNIST dataset and label metadata.

The notebook then finds the repository root with `find_project_root`.

```python
def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "README.md").is_file() and (candidate / "flake.nix").is_file():
            return candidate
```

Jupyter often starts a notebook kernel inside the `notebooks/` directory instead of the repository root. The function checks that directory and each parent directory until it finds the project markers `README.md` and `flake.nix`. This avoids writing outputs inside `notebooks/` accidentally.

The same cell defines fixed project settings:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `AUDIT_SIZE` | 28 | Native stored size used only for the audit. |
| Training size | 224 × 224 | Images will be resized for ResNet-18 training later. |
| `SEED` | 2026 | Random seed used to make the 1% subset reproducible. |
| `SUBSET_FRACTION` | 0.01 | Fraction of the training split used for the limited-data condition. |

### 2. Define helper functions

#### `counts_and_prevalence`

```python
def counts_and_prevalence(labels):
```

This function flattens the label array, counts examples in every class, and calculates the percentage of the split belonging to each class:

\[
\text{prevalence}(c) = \frac{\text{number of images in class } c}{\text{number of images in the split}} \times 100
\]

It returns one record per class containing class index, class name, count, and prevalence percentage.

#### `stratified_subset_indices`

```python
def stratified_subset_indices(labels, fraction, seed):
```

This function creates the fixed limited-data subset.

1. It sets the target size to `round(N × fraction)`, where `N` is the number of training images.
2. It calculates each class's proportional allocation.
3. It initially takes the floor of each allocation.
4. It distributes any remaining images to the classes with the largest fractional remainders. This is called **largest-remainder allocation**.
5. It uses NumPy's seeded random generator to choose the allocated number of indices from each class without replacement.
6. It sorts the selected indices before saving them.

The output contains exactly 120 training indices while keeping the class distribution as close as possible to the full training distribution.

### 3. Download and load the official splits

The second code cell runs:

```python
BloodMNIST(split=split, size=AUDIT_SIZE, root=DATA_ROOT, download=True)
```

for the official `train`, `val`, and `test` splits.

- `download=True` downloads the dataset to `.data/medmnist/` only if it is not already present.
- The `.data/` directory is ignored by Git, so the dataset is not committed.
- The official split is retained; no new train/validation/test split is created.

The cell confirms the expected totals:

| Split | Expected image count |
| --- | ---: |
| Training | 11,959 |
| Validation | 1,712 |
| Test | 3,421 |

The cell checks that training images are RGB and that the training labels contain exactly the documented eight BloodMNIST classes.

> **Test-set protection:** the notebook calls `len(test)` to record the test-set total but does not read, count, display, or otherwise inspect test labels.

### 4. Create the required counts table

The next cell applies `counts_and_prevalence` to the training and validation labels and builds a Pandas table with:

- class name;
- training count and prevalence;
- validation count and prevalence.

A total row confirms that prevalence sums to 100% in each split. This table is the source for the required progress-report counts table.

### 5. Inspect one labelled image per class

The following cell finds the first training image from each class and displays it in a 2 × 4 grid.

This is a basic visual audit. It helps confirm that:

- images load correctly;
- labels map to the intended blood-cell names;
- images appear plausible before expensive experiments begin.

These displayed images come only from the training split.

### 6. Create and review the 1% subset

The next cell creates the subset:

```python
subset_indices = stratified_subset_indices(train.labels, SUBSET_FRACTION, SEED)
```

It then calculates a second class table for those selected images. Review this table before saving. The expected size is 120 images.

Every 1%-data experiment must load the **same saved index file**, rather than creating a new random subset for each seed or model. Model seeds change initialization and training randomness; they must not change which examples are available.

### 7. Save reproducibility artifacts

The final code cell writes the following artifacts after you have reviewed the tables.

| Output | Purpose | Git status |
| --- | --- | --- |
| `artifacts/data_audit/official_split_counts.md` | Report-ready train/validation counts and test total. | Commit |
| `splits/bloodmnist_train_1pct_seed2026.npz` | Fixed 1% selected training indices and metadata. | Commit |
| `artifacts/data_audit/train_class_counts.csv` | Machine-readable training counts. | Ignored generated output |
| `artifacts/data_audit/val_class_counts.csv` | Machine-readable validation counts. | Ignored generated output |
| `artifacts/data_audit/train_1pct_class_counts.csv` | Machine-readable 1% subset counts. | Ignored generated output |
| `artifacts/data_audit/labelled_training_samples.png` | Saved labelled sample grid. | Ignored generated output |
| `artifacts/data_audit/audit_summary.json` | Dataset, resolution, split, and subset metadata. | Ignored generated output |

The subset `.npz` contains:

- `indices`: selected positions in the official training split;
- `seed`: `2026`;
- `fraction`: `0.01`;
- `source_split`: `train`;
- `source_size`: `11959`.

## After running

1. Verify the official split totals and count tables.
2. Verify that the 1% subset contains 120 images with all eight classes represented.
3. Commit the notebook, this documentation file, the report-ready counts table, and the fixed subset file.
4. Update [`../progress_tracker.md`](../progress_tracker.md) to mark the completed data-audit tasks.
5. Use the saved index file in later linear-probe, fine-tuning, and random-initialization runs at the 1% data size.

Do not commit `.data/` or `.venv/`. Do not use the test labels until model choices have been frozen for the final evaluation.

## Relation to later experiments

This notebook does not train a model. Its outputs enable the next stages:

1. Run the majority-class baseline.
2. Run the frozen ResNet-18 linear probe, caching frozen features for no-augmentation runs.
3. Compare linear probe, full fine-tuning, and random initialization at both 1% and 100% training sizes.
4. Select the best strategy using validation macro-F1.
5. Perform the required augmentation comparison.
6. Read the test set only once after all decisions are final.
