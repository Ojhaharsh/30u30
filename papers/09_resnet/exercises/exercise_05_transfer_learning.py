"""
Exercise 5: Transfer Learning with ResNet
=========================================

Goal: Fine-tune pretrained ResNet on custom dataset.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_pretrained_resnet(num_classes, freeze_backbone=True):
    """Load pretrained ResNet and modify for new task."""
    # TODO 1: Load pretrained ResNet-18
    model = None  # TODO: models.resnet18(pretrained=True)
    
    # TODO 2: Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            # TODO: param.requires_grad = False
            pass
    
    # TODO 3: Replace final FC layer
    # model.fc = nn.Linear(512, num_classes)
    
    return model


def fine_tune(model, train_loader, val_loader, epochs=10):
    """Fine-tune the model."""
    # TODO 4: Training loop with frozen backbone
    pass


if __name__ == "__main__":
    print(__doc__)
    # model = get_pretrained_resnet(10)
