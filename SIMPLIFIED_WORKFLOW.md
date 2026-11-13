# Simple training workflow

The first deep-learning experiment is intentionally written as five small Python files at the repository root. Read them in this order:

1. `config.py` — one plain `CONFIG` dictionary. Change epochs, batch size, learning rate, seed, or output folder here.
2. `data.py` — gets approved BloodMNIST training and validation loaders from `bloodmnist_baselines`. It does not create or load a test split.
3. `model.py` — creates pretrained ResNet-18 and changes its final layer to predict the 8 BloodMNIST classes.
4. `train.py` — direct training and validation loops. `train_one_epoch` updates weights; `validate` only measures validation loss, accuracy, and macro-F1.
5. `run_experiment.py` — the top-to-bottom program. It checks for a T4 GPU, gets data, builds the model, repeats train/validate for each epoch, then saves outputs.

Run only through `notebooks/colab_train_resnet18.ipynb` on a Google Colab NVIDIA T4 runtime. The notebook clones the repository, checks out its pinned commit, installs packages, and invokes the root training script with `PYTHONPATH=src`.

Results are written under `outputs/<UTC timestamp>_bloodmnist_resnet18_1pct/`:

- `run_metadata.json` — configuration, provider provenance, GPU, and Git commit;
- `history.csv` — one row per epoch;
- `learning_curves.png` — loss and validation macro-F1 curves;
- `best_model.pt` — model selected only by validation macro-F1;
- `last_model.pt` — final epoch model;
- `summary.json` — best epoch, best validation macro-F1, and runtime.

The locked test split is never loaded or evaluated. No local experiment execution is permitted; run the notebook in Colab only after explicit authorization.
