"""Edit these values to define one BloodMNIST ResNet-18 experiment.

Do not put the test split here. It is intentionally not available to this program.
"""

CONFIG = {
    "run_name": "bloodmnist_resnet18_1pct",
    "seed": 2026,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 2,
    "image_size": 224,
    "num_classes": 8,
    "use_fixed_1pct_subset": True,
    "audit_manifest_path": "artifacts/data_audit/audit_manifest.json",
    "subset_manifest_path": "splits/bloodmnist_train_1pct_seed2026.npz",
    "output_root": "outputs",
}
