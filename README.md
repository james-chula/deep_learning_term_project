# Blood Cell Classification with ResNet-18

## Jupyter development environment

This repository uses a Nix flake and direnv to provide a project-scoped Python/Jupyter runtime. The environment includes Python 3.12, PyTorch, torchvision, JupyterLab, and common analysis packages.

### One-time setup

From the repository root:

```bash
direnv allow
./scripts/setup-runtime
```

Afterward, entering the repository automatically loads the flake shell. Start JupyterLab with:

```bash
./scripts/jupyter
```

Open a notebook and select the **Python (BloodMNIST ResNet)** kernel. The repository-local `.jupyter/` directory (intentionally ignored by Git) stores its kernel, JupyterLab settings, extensions/data, and transient server runtime files. Your global Jupyter settings in `~/.jupyter/` are not used when Jupyter is started through `./scripts/jupyter`.

### Data audit and fixed 1% subset

Generate the required official-split counts table and reproducible class-stratified 1% training subset with:

```bash
./scripts/setup-runtime
./scripts/jupyter
```

Then open [`notebooks/01_data_audit_and_subset.ipynb`](notebooks/01_data_audit_and_subset.ipynb) and select the **Python (BloodMNIST ResNet)** kernel. Run its cells in order; the download happens from the notebook.

The notebook installs no packages itself. `./scripts/setup-runtime` installs the local MedMNIST dependency if necessary. The notebook downloads the compact official BloodMNIST source file into the ignored `.data/` directory. Its split membership and labels are the same as the 224×224 release; training code will resize images to 224×224. Versioned audit outputs are written to `artifacts/data_audit/`; the selected indices, allocation seed (2026), and subset metadata are saved in `splits/bloodmnist_train_1pct_seed2026.npz`. The notebook does not inspect test labels and records only the test-set total.

### GPU note

The flake provides the project runtime and installs PyTorch. CUDA availability depends on the host system and Nix CUDA configuration. Verify the selected runtime in a notebook before GPU training:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
