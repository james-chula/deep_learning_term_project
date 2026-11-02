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

### GPU note

The flake provides the project runtime and installs PyTorch. CUDA availability depends on the host system and Nix CUDA configuration. Verify the selected runtime in a notebook before GPU training:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
