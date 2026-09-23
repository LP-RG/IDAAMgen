from __future__ import annotations

import gc
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
trained_models_path = os.path.join(ROOT_DIR, "trained_models/")
SRC_PATH = os.path.join(ROOT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
import models.model_factory as factory
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME_ALIASES = {
    "resnet": "resnet50",
    "resnet-50": "resnet50",
    "resnet-20": "resnet20_cifar",
    "resnext": "resnext50_32x4d",
    "resnext50": "resnext50_32x4d",
    "mobilenet": "mobilenet_v2",
    "mobilenetv2": "mobilenet_v2",
    "convnext": "convnext_tiny",
    "convnext-t": "convnext_tiny",
    "vgg": "vgg16_bn",
    "vgg16": "vgg16_bn",
}

MODEL_IMAGE_SHAPES = {
    "resnet50":        (3, 224, 224),
    "resnet20_cifar":  (3, 32, 32),
    "resnext50_32x4d": (3, 224, 224),
    "mobilenet_v2":     (3, 224, 224),
    "convnext_tiny":   (3, 224, 224),
    "vgg16_bn":        (3, 224, 224),
}

MODEL_CONFIG_BUILDERS = {
    "resnet50":        factory.resnet50_config,
    "resnet20_cifar":  factory.resnet20_cifar_config,
    "resnext50_32x4d": factory.resnext50_32x4d_config,
    "mobilenet_v2":     factory.mobilenet_v2_config,
    "convnext_tiny":   factory.convnext_tiny_config,
    "vgg16_bn":        factory.vgg16_bn_config,
}

train_loader = None
test_loader = None
_classes = None
dataset_name = None


def normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    return MODEL_NAME_ALIASES.get(name, name)


def build_model(
    model_name: str,
    conv_type: int,
    bit_width: int,
    signed: bool,
    zone: bool,
    multiplier_matrix=None,
    num_classes: int = 10,
    shift_bits: int = 0,
    dataset_name: Optional[str] = None, 
) -> nn.Sequential:
    norm_name = normalize_model_name(model_name)
    if norm_name not in MODEL_CONFIG_BUILDERS:
        raise ValueError(
            f"Modello '{model_name}' (normalizzato in '{norm_name}') non supportato nel subset SOTA. "
            f"Modelli disponibili: {list(MODEL_CONFIG_BUILDERS.keys())}"
        )

    is_cifar = dataset_name is not None and "cifar" in dataset_name.lower()

    config_fn = MODEL_CONFIG_BUILDERS[norm_name]
    
    import inspect
    sig = inspect.signature(config_fn)
    
    config_kwargs = {"num_classes": num_classes}
    if "is_cifar" in sig.parameters:
        config_kwargs["is_cifar"] = is_cifar

    layer_specs = config_fn(**config_kwargs)

    conv_defaults = {
        "conv_type": conv_type,
        "bit_width": bit_width,
        "signed": signed,
        "zone": zone,
        "multiplier_matrix": multiplier_matrix,
        "shift_bits": shift_bits,
    }

    model = factory.build_model(layer_specs, conv_defaults=conv_defaults)
    return model.to(device)

import random 

def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True, warn_only=True)

def clean_gpu(model=None, optimizer=None, scheduler=None):
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    if scheduler is not None:
        del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()