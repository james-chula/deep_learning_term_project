"""Get approved BloodMNIST training and validation loaders.

The shared provider constructs only official train and validation data. The test
set is not constructed, exposed, or evaluated by this workflow.
"""
import json
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from bloodmnist_baselines import build_provider_datasets


def make_transforms(image_size):
    """Resize official 28px images exactly once to 224px."""
    if image_size != 224:
        raise ValueError("The approved workflow resizes 28px BloodMNIST images to 224px.")
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]),
    )


def get_data_loaders(config):
    """Return train/validation loaders and the provider's locked-test provenance."""
    train_transform, validation_transform = make_transforms(config["image_size"])
    audit_path = Path(config["audit_manifest_path"])
    subset_path = None
    if config["use_fixed_1pct_subset"]:
        if not audit_path.is_file():
            raise FileNotFoundError(f"Audit manifest not found: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        declared_subset = audit.get("subset_manifest")
        if not isinstance(declared_subset, str) or not declared_subset.strip():
            raise ValueError("Fixed 1% mode requires subset_manifest in the audit manifest; no full-data fallback exists.")
        # Audit paths are repository-root-relative, matching audit_manifest_path.
        subset_path = Path(declared_subset)
        configured_override = config.get("subset_manifest_path")
        if configured_override is not None and Path(configured_override) != subset_path:
            raise ValueError("Configured subset manifest override differs from the audited subset manifest.")
    datasets = build_provider_datasets(
        audit_manifest_path=audit_path,
        image_size=config["image_size"],
        train_transform=train_transform,
        val_transform=validation_transform,
        subset_manifest_path=subset_path,
    )
    metadata = datasets.metadata
    if metadata.get("test_policy") != "locked_not_loaded":
        raise ValueError("Provider must keep the test split locked and unloaded.")
    if metadata.get("archive_size") != 28 or metadata.get("resize_to") != 224:
        raise ValueError("Expected official 28px data resized once to 224px.")
    if subset_path and not metadata.get("subset_manifest_sha256"):
        raise ValueError("Provider did not apply the requested fixed 1% training subset.")
    return (
        DataLoader(datasets.train, batch_size=config["batch_size"], shuffle=True,
                   num_workers=config["num_workers"], pin_memory=True),
        DataLoader(datasets.val, batch_size=config["batch_size"], shuffle=False,
                   num_workers=config["num_workers"], pin_memory=True),
        metadata,
    )
