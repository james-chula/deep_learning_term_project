"""Create the pretrained image classifier used in this project."""


def create_model(num_classes=8):
    """Load ImageNet-pretrained ResNet-18 and replace its final 8-class layer."""
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
