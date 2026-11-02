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

Open a notebook and select the **Python (BloodMNIST ResNet)** kernel. Its kernelspec is stored under `.jupyter/`, which is intentionally ignored by Git.

### GPU note

The flake provides the project runtime and installs PyTorch. CUDA availability depends on the host system and Nix CUDA configuration. Verify the selected runtime in a notebook before GPU training:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
