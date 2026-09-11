from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import modules.convolution as cc
import models.resnet8 as resnet8
import models.resnet20 as resnet20
import models.lenet5 as lenet5
import models.vgg16 as vgg16
import models.alexnet_cifar10 as alexnet_cifar10
import models.resnet56 as resnet56
import sys
import gc
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
trained_models_path = os.path.join(ROOT_DIR, "trained_models/")
SRC_PATH = os.path.join(ROOT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

device = "cuda"

MODEL_NAME_ALIASES = {
    "resnet20": "resnet",
    "lenet": "lenet5",
}

MODEL_IMAGE_SHAPES = {
    "lenet5":          (1, 32, 32),
    "resnet":          (3, 32, 32),
    "resnet8":         (3, 32, 32),
    "vgg16":           (3, 32, 32),
    "alexnet_cifar10": (3, 32, 32),
    "resnet56":        (3, 32, 32),
}

MODEL_FACTORIES = {
    "resnet": resnet20.ResNet20,
    "lenet5": lenet5.LeNet5,
    "vgg16": vgg16.VGG16,
    "alexnet_cifar10": alexnet_cifar10.AlexNetCIFAR10,
    "resnet56": resnet56.ResNet56_CIFAR100,
    "resnet8": resnet8.ResNet8,
}

train_loader = None
test_loader = None
_classes = None
dataset_name = None


def normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    return MODEL_NAME_ALIASES.get(name, name)



def build_model(model_name: str, conv_type: int, bit_width: int, signed: bool, zone: bool,
                multiplier_matrix: str | list[str] = None, num_classes: int = 10):
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Model '{model_name}' not supported.")
    return MODEL_FACTORIES[model_name](
        multiplier_matrix,
        num_classes=num_classes,
        conv_type=conv_type,
        bit_width=bit_width,
        signed=signed,
        zone=zone,
    ).to(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def clean_gpu(model=None, optimizer=None, scheduler=None):
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
