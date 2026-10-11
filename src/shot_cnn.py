"""
src/shot_cnn.py
===============
Appearance branch of the shot-type recogniser.

A ResNet-18 (ImageNet weights) fine-tuned on the batsman crops in
data/shot_dataset. The dataset images are grayscale crops of a batsman at
the moment of the shot, so at inference time video frames are converted to
grayscale and cropped around the detected batsman before being classified.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 192
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


def train_transform(size: int = IMG_SIZE):
    # Random crops with a wide aspect range mimic the variable tightness of
    # the pose-based crop used on video frames.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.6, 1.7)),
        transforms.RandomHorizontalFlip(),          # left/right-handers share labels
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.35, contrast=0.35),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.6))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.inference_mode()
def predict_proba(model: nn.Module, images: list[Image.Image], size: int = IMG_SIZE,
                  batch_size: int = 32) -> np.ndarray:
    """Class probabilities with horizontal-flip test-time augmentation."""
    model.eval()
    tf = eval_transform(size)
    out = []
    for s in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[s:s + batch_size]])
        p = torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[3])), 1)
        out.append((p / 2).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))
