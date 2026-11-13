"""Model and transform factories for the fixed pretrained ResNet-18 condition."""
from __future__ import annotations

def build_transforms():
    """Return deterministic validation and conservative training transforms.

    The provider applies these supplied transforms to official 28px data; both paths explicitly resize to 224px. Random augmentation is training-only.
    """
    from torchvision import transforms
    normalize = transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    return (transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
            transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize]))

def build_resnet18(num_classes: int = 8):
    """Create ImageNet-pretrained ResNet-18 with a project-specific classifier."""
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
