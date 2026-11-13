"""Edit this one dictionary to select a validation-only BloodMNIST run.

Allowed modes: majority, linear_probe, finetune. Test data are intentionally not
represented in this configuration or anywhere in the execution workflow.
"""

CONFIG = {
    "mode": "finetune",  # majority | linear_probe | finetune
    "run_name": "bloodmnist_finetune_1pct",
    "seed": 2026,
    "epochs": 10,  # used by linear_probe and finetune; ignored by majority
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 2,
    "image_size": 224,
    "num_classes": 8,
    "use_fixed_1pct_subset": True,
    "audit_manifest_path": "artifacts/data_audit/audit_manifest.json",
    "output_root": "outputs",
}
