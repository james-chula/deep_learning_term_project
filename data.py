"""Get the approved BloodMNIST train and validation loaders.

The provider package owns data download, audit checks, official split membership,
and the fixed training subset. This file never constructs or loads a test split.
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from bloodmnist_baselines import build_provider_datasets


def make_transforms():
    """Resize 28px images to 224px; use augmentation only while training."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, validation_transform


def get_data_loaders(config):
    """Return train_loader, validation_loader, and provider metadata.

    Raises an error if the approved provider does not prove the locked-test policy.
    """
    train_transform, validation_transform = make_transforms()
    subset_path = config["subset_manifest_path"] if config["use_fixed_1pct_subset"] else None
    datasets = build_provider_datasets(
        audit_manifest_path=Path(config["audit_manifest_path"]),
        image_size=config["image_size"],
        train_transform=train_transform,
        val_transform=validation_transform,
        subset_manifest_path=Path(subset_path) if subset_path else None,
    )
    metadata = datasets.metadata
    if metadata.get("test_policy") != "locked_not_loaded":
        raise ValueError("Provider must keep the test split locked and unloaded.")
    if metadata.get("archive_size") != 28 or metadata.get("resize_to") != 224:
        raise ValueError("Expected official 28px data resized to 224px.")

    train_loader = DataLoader(
        datasets.train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    validation_loader = DataLoader(
        datasets.val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    return train_loader, validation_loader, metadata
