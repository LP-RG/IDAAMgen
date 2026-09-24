import enum
from typing import Callable, Optional

import torch
import torch.nn as nn
import modules.convolution as cc  # Modulo custom con Conv2d_custom


def make_conv_builder(conv_defaults: dict) -> Callable[..., nn.Module]:
    def _build(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1, name="", **overrides):
        params = {**conv_defaults, **overrides}
        return cc.Conv2d_custom(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, groups=groups, name=name, **params,
        )
    return _build



class BasicBlock(nn.Module):
    """ResNet CIFAR (ResNet-20): Basic Block 3x3 -> 3x3."""
    expansion = 1

    def __init__(self, in_channels, out_channels, conv_builder, stride=1, name="0"):
        super().__init__()
        self.conv1 = conv_builder(in_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=1, bias=False, name=name + "_1")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_builder(out_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False, name=name + "_2")
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv_builder(in_channels, out_channels, kernel_size=1, stride=stride,
                             padding=0, bias=False, name=name + "_s"),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet-50: Bottleneck 1x1 -> 3x3 -> 1x1 (expansion=4)."""
    expansion = 4

    def __init__(self, in_channels, out_channels, conv_builder, stride=1, name="0"):
        super().__init__()
        self.conv1 = conv_builder(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False, name=name + "_1")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_builder(out_channels, out_channels, kernel_size=3, stride=stride,
                                   padding=1, bias=False, name=name + "_2")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv_builder(out_channels, out_channels * self.expansion, kernel_size=1,
                                   stride=1, padding=0, bias=False, name=name + "_3")
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv_builder(in_channels, out_channels * self.expansion, kernel_size=1,
                             stride=stride, padding=0, bias=False, name=name + "_s"),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNeXtBlock(nn.Module):
    """ResNeXt-50 (32x4d): Bottleneck con Conv 3x3 Raggruppata."""
    expansion = 4

    def __init__(self, in_channels, out_channels, conv_builder, stride=1,
                 cardinality=32, base_width=4, name="0"):
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * cardinality
        self.conv1 = conv_builder(in_channels, width, kernel_size=1, stride=1, padding=0,
                                   bias=False, name=name + "_1")
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv_builder(width, width, kernel_size=3, stride=stride, padding=1,
                                   bias=False, groups=cardinality, name=name + "_2")
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv_builder(width, out_channels * self.expansion, kernel_size=1, stride=1,
                                   padding=0, bias=False, name=name + "_3")
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv_builder(in_channels, out_channels * self.expansion, kernel_size=1,
                             stride=stride, padding=0, bias=False, name=name + "_s"),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class InvertedResidualBlock(nn.Module):
    """MobileNetV2: Expand 1x1 -> Depthwise 3x3 -> Project 1x1 Linear."""

    def __init__(self, in_channels, out_channels, conv_builder, stride=1, expand_ratio=6, name="0"):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers += [
                conv_builder(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
                             bias=False, name=name + "_expand"),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            conv_builder(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                         bias=False, groups=hidden_dim, name=name + "_dw"),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            conv_builder(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0,
                         bias=False, name=name + "_project"),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


class LayerNorm2d(nn.Module):
    """LayerNorm per tensori 4D (N, C, H, W) in formato channels_first."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW


class ConvNeXtBlock(nn.Module):
    """ConvNeXt: DW 7x7 -> LayerNorm -> MLP Pointwise (4x) con GELU."""

    def __init__(self, dim, conv_builder, name="0", layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = conv_builder(dim, dim, kernel_size=7, stride=1, padding=3, bias=True,
                                    groups=dim, name=name + "_dw")
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  
        return residual + x


# ---------------------------------------------------------------------------
# Stage Builders
# ---------------------------------------------------------------------------

def build_basic_block_stage(params: dict, conv_builder: Callable) -> nn.Sequential:
    in_channels, out_channels = params["in_channels"], params["out_channels"]
    num_blocks, stride = params["num_blocks"], params.get("stride", 1)
    base_name = params.get("name", "stage")

    layers = [BasicBlock(in_channels, out_channels, conv_builder, stride=stride, name=f"{base_name}a")]
    for i in range(1, num_blocks):
        suffix = chr(ord("a") + i)
        layers.append(BasicBlock(out_channels * BasicBlock.expansion, out_channels, conv_builder,
                                  stride=1, name=f"{base_name}{suffix}"))
    return nn.Sequential(*layers)


def build_bottleneck_stage(params: dict, conv_builder: Callable) -> nn.Sequential:
    in_channels, out_channels = params["in_channels"], params["out_channels"]
    num_blocks, stride = params["num_blocks"], params.get("stride", 1)
    base_name = params.get("name", "stage")

    layers = [Bottleneck(in_channels, out_channels, conv_builder, stride=stride, name=f"{base_name}a")]
    for i in range(1, num_blocks):
        suffix = chr(ord("a") + i)
        layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels, conv_builder,
                                  stride=1, name=f"{base_name}{suffix}"))
    return nn.Sequential(*layers)


def build_resnext_stage(params: dict, conv_builder: Callable) -> nn.Sequential:
    in_channels, out_channels = params["in_channels"], params["out_channels"]
    num_blocks, stride = params["num_blocks"], params.get("stride", 1)
    cardinality, base_width = params.get("cardinality", 32), params.get("base_width", 4)
    base_name = params.get("name", "stage")

    layers = [ResNeXtBlock(in_channels, out_channels, conv_builder, stride=stride,
                            cardinality=cardinality, base_width=base_width, name=f"{base_name}a")]
    for i in range(1, num_blocks):
        suffix = chr(ord("a") + i)
        layers.append(ResNeXtBlock(out_channels * ResNeXtBlock.expansion, out_channels, conv_builder,
                                    stride=1, cardinality=cardinality, base_width=base_width,
                                    name=f"{base_name}{suffix}"))
    return nn.Sequential(*layers)


def build_inverted_residual_stage(params: dict, conv_builder: Callable) -> nn.Sequential:
    in_channels, out_channels = params["in_channels"], params["out_channels"]
    num_blocks, stride = params["num_blocks"], params.get("stride", 1)
    expand_ratio = params.get("expand_ratio", 6)
    base_name = params.get("name", "stage")

    layers = [InvertedResidualBlock(in_channels, out_channels, conv_builder, stride=stride,
                                     expand_ratio=expand_ratio, name=f"{base_name}a")]
    for i in range(1, num_blocks):
        suffix = chr(ord("a") + i)
        layers.append(InvertedResidualBlock(out_channels, out_channels, conv_builder, stride=1,
                                             expand_ratio=expand_ratio, name=f"{base_name}{suffix}"))
    return nn.Sequential(*layers)


def build_convnext_stage(params: dict, conv_builder: Callable) -> nn.Sequential:
    in_channels, out_channels = params["in_channels"], params["out_channels"]
    num_blocks, downsample = params["num_blocks"], params.get("downsample", True)
    name = params.get("name", "stage")

    layers = []
    if downsample:
        layers.append(LayerNorm2d(in_channels))
        layers.append(conv_builder(in_channels, out_channels, kernel_size=2, stride=2,
                                    padding=0, bias=True, name=name + "_ds"))
    for i in range(num_blocks):
        layers.append(ConvNeXtBlock(out_channels, conv_builder, name=f"{name}_b{i}"))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Layer Registry & Factory
# ---------------------------------------------------------------------------

class LayerType(enum.Enum):
    CONV = "conv"
    BATCHNORM = "bn"
    RELU = "relu"
    RELU6 = "relu6"
    MAXPOOL = "maxpool"
    ADAPTIVE_AVG_POOL = "avgpool"
    FLATTEN = "flatten"
    DROPOUT = "dropout"
    LINEAR = "linear"
    LAYERNORM2D = "layernorm2d"
    BASIC_BLOCK_STAGE = "basic_block_stage"
    BOTTLENECK_STAGE = "bottleneck_stage"
    RESNEXT_STAGE = "resnext_stage"
    INVERTED_RESIDUAL_STAGE = "inverted_residual_stage"
    CONVNEXT_STAGE = "convnext_stage"


LAYER_REGISTRY: dict[LayerType, Callable[[dict, Callable], nn.Module]] = {
    LayerType.CONV: lambda p, conv_builder: conv_builder(**p),
    LayerType.BATCHNORM: lambda p, _: nn.BatchNorm2d(**p),
    LayerType.RELU: lambda p, _: nn.ReLU(**p),
    LayerType.RELU6: lambda p, _: nn.ReLU6(**p),
    LayerType.MAXPOOL: lambda p, _: nn.MaxPool2d(**p),
    LayerType.ADAPTIVE_AVG_POOL: lambda p, _: nn.AdaptiveAvgPool2d(**p),
    LayerType.FLATTEN: lambda p, _: nn.Flatten(**p),
    LayerType.DROPOUT: lambda p, _: nn.Dropout(**p),
    LayerType.LINEAR: lambda p, _: nn.Linear(**p),
    LayerType.LAYERNORM2D: lambda p, _: LayerNorm2d(**p),
    LayerType.BASIC_BLOCK_STAGE: build_basic_block_stage,
    LayerType.BOTTLENECK_STAGE: build_bottleneck_stage,
    LayerType.RESNEXT_STAGE: build_resnext_stage,
    LayerType.INVERTED_RESIDUAL_STAGE: build_inverted_residual_stage,
    LayerType.CONVNEXT_STAGE: build_convnext_stage,
}


def _build_sequential(layer_specs: list[dict], conv_builder: Callable) -> nn.Sequential:
    layers = []
    for spec in layer_specs:
        layer_type = spec["type"]
        if layer_type not in LAYER_REGISTRY:
            raise ValueError(f"Tipo di layer non riconosciuto: {layer_type}")
        builder = LAYER_REGISTRY[layer_type]
        layers.append(builder(spec.get("params", {}), conv_builder))
    return nn.Sequential(*layers)


def build_model(layer_specs: list[dict], conv_defaults: Optional[dict] = None) -> nn.Sequential:
    conv_builder = make_conv_builder(conv_defaults or {})
    return _build_sequential(layer_specs, conv_builder)


# ===========================================================================
# Subset SOTA Configurations
# ===========================================================================

# --- 1. ResNet-50 (ImageNet) & ResNet-20 (CIFAR) ---
def resnet50_config(num_classes: int = 100, in_channels: int = 3, is_cifar: bool = True) -> list[dict]:
    """ResNet-50 configurabile per ImageNet (224x224) o CIFAR (32x32)."""
    blocks_per_stage = [3, 4, 6, 3]
    widths = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    if is_cifar:
        specs = [
            {"type": LayerType.CONV, "params": {"in_channels": in_channels, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False, "name": "stem"}},
            {"type": LayerType.BATCHNORM, "params": {"num_features": 64}},
            {"type": LayerType.RELU, "params": {"inplace": True}},
        ]
    else:
        specs = [
            {"type": LayerType.CONV, "params": {"in_channels": in_channels, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3, "bias": False, "name": "stem"}},
            {"type": LayerType.BATCHNORM, "params": {"num_features": 64}},
            {"type": LayerType.RELU, "params": {"inplace": True}},
            {"type": LayerType.MAXPOOL, "params": {"kernel_size": 3, "stride": 2, "padding": 1}},
        ]

    in_ch = 64
    for i, (w, nblocks, stride) in enumerate(zip(widths, blocks_per_stage, strides)):
        specs.append({"type": LayerType.BOTTLENECK_STAGE, "params": {
            "in_channels": in_ch, "out_channels": w, "num_blocks": nblocks, "stride": stride, "name": f"layer{i + 1}"
        }})
        in_ch = w * Bottleneck.expansion

    specs += [
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.LINEAR, "params": {"in_features": in_ch, "out_features": num_classes}},
    ]
    return specs


def resnet20_cifar_config(num_classes: int = 10) -> list[dict]:
    """ResNet-20 per CIFAR-10 (BasicBlock)."""
    return [
        {"type": LayerType.CONV, "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False, "name": "stem"}},
        {"type": LayerType.BATCHNORM, "params": {"num_features": 16}},
        {"type": LayerType.RELU, "params": {"inplace": True}},
        {"type": LayerType.BASIC_BLOCK_STAGE, "params": {"in_channels": 16, "out_channels": 16, "num_blocks": 3, "stride": 1, "name": "stage1"}},
        {"type": LayerType.BASIC_BLOCK_STAGE, "params": {"in_channels": 16, "out_channels": 32, "num_blocks": 3, "stride": 2, "name": "stage2"}},
        {"type": LayerType.BASIC_BLOCK_STAGE, "params": {"in_channels": 32, "out_channels": 64, "num_blocks": 3, "stride": 2, "name": "stage3"}},
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.LINEAR, "params": {"in_features": 64, "out_features": num_classes}},
    ]


# --- 2. ResNeXt-50 (32x4d) ---

def resnext50_32x4d_config(num_classes: int = 1000, in_channels: int = 3) -> list[dict]:
    """ResNeXt-50 32x4d per ImageNet."""
    blocks_per_stage = [3, 4, 6, 3]
    widths = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    specs = [
        {"type": LayerType.CONV, "params": {"in_channels": in_channels, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3, "bias": False, "name": "stem"}},
        {"type": LayerType.BATCHNORM, "params": {"num_features": 64}},
        {"type": LayerType.RELU, "params": {"inplace": True}},
        {"type": LayerType.MAXPOOL, "params": {"kernel_size": 3, "stride": 2, "padding": 1}},
    ]

    in_ch = 64
    for i, (w, nblocks, stride) in enumerate(zip(widths, blocks_per_stage, strides)):
        specs.append({"type": LayerType.RESNEXT_STAGE, "params": {
            "in_channels": in_ch, "out_channels": w, "num_blocks": nblocks, "stride": stride,
            "cardinality": 32, "base_width": 4, "name": f"layer{i + 1}"
        }})
        in_ch = w * ResNeXtBlock.expansion

    specs += [
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.LINEAR, "params": {"in_features": in_ch, "out_features": num_classes}},
    ]
    return specs


# --- 3. MobileNetV2 ---

def mobilenet_v2_config(num_classes: int = 1000, width_mult: float = 1.0) -> list[dict]:
    """MobileNetV2 Standard."""
    def c(ch): return max(8, int(ch * width_mult))

    # [t, c, n, s] -> expand_ratio, output_channels, num_blocks, stride
    inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    in_channels = c(32)
    specs = [
        {"type": LayerType.CONV, "params": {"in_channels": 3, "out_channels": in_channels, "kernel_size": 3, "stride": 2, "padding": 1, "bias": False, "name": "stem"}},
        {"type": LayerType.BATCHNORM, "params": {"num_features": in_channels}},
        {"type": LayerType.RELU6, "params": {"inplace": True}},
    ]

    for i, (t, c_out, n, s) in enumerate(inverted_residual_setting):
        out_channels = c(c_out)
        specs.append({"type": LayerType.INVERTED_RESIDUAL_STAGE, "params": {
            "in_channels": in_channels, "out_channels": out_channels, "num_blocks": n,
            "stride": s, "expand_ratio": t, "name": f"stage{i + 1}"
        }})
        in_channels = out_channels

    last_conv_channels = c(1280)
    specs += [
        {"type": LayerType.CONV, "params": {"in_channels": in_channels, "out_channels": last_conv_channels, "kernel_size": 1, "stride": 1, "padding": 0, "bias": False, "name": "conv_head"}},
        {"type": LayerType.BATCHNORM, "params": {"num_features": last_conv_channels}},
        {"type": LayerType.RELU6, "params": {"inplace": True}},
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.DROPOUT, "params": {"p": 0.2}},
        {"type": LayerType.LINEAR, "params": {"in_features": last_conv_channels, "out_features": num_classes}},
    ]
    return specs


# --- 4. ConvNeXt-Tiny ---

def convnext_tiny_config(num_classes: int = 1000) -> list[dict]:
    """ConvNeXt-Tiny (4 stages: 3, 3, 9, 3 blocks; channels: 96, 192, 384, 768)."""
    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]

    # Patchify Stem
    specs = [
        {"type": LayerType.CONV, "params": {"in_channels": 3, "out_channels": dims[0], "kernel_size": 4, "stride": 4, "padding": 0, "bias": True, "name": "stem"}},
        {"type": LayerType.LAYERNORM2D, "params": {"num_channels": dims[0]}},
    ]

    # Stem stage blocks
    specs.append({"type": LayerType.CONVNEXT_STAGE, "params": {
        "in_channels": dims[0], "out_channels": dims[0], "num_blocks": depths[0], "downsample": False, "name": "stage1"
    }})

    # Downsampling stages
    for i in range(1, 4):
        specs.append({"type": LayerType.CONVNEXT_STAGE, "params": {
            "in_channels": dims[i - 1], "out_channels": dims[i], "num_blocks": depths[i], "downsample": True, "name": f"stage{i + 1}"
        }})

    specs += [
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.LAYERNORM2D, "params": {"num_channels": dims[-1]}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.LINEAR, "params": {"in_features": dims[-1], "out_features": num_classes}},
    ]
    return specs


# --- 5. VGG-16 (con BatchNorm) ---

def vgg16_bn_config(num_classes: int = 1000, in_channels: int = 3) -> list[dict]:
    """VGG-16 con BatchNorm."""
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    specs = []
    ch = in_channels

    for i, v in enumerate(cfg):
        if v == "M":
            specs.append({"type": LayerType.MAXPOOL, "params": {"kernel_size": 2, "stride": 2}})
            continue
        specs.append({"type": LayerType.CONV, "params": {"in_channels": ch, "out_channels": v, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False, "name": f"conv{i}"}})
        specs.append({"type": LayerType.BATCHNORM, "params": {"num_features": v}})
        specs.append({"type": LayerType.RELU, "params": {"inplace": True}})
        ch = v

    specs += [
        {"type": LayerType.ADAPTIVE_AVG_POOL, "params": {"output_size": 1}},
        {"type": LayerType.FLATTEN, "params": {}},
        {"type": LayerType.LINEAR, "params": {"in_features": ch, "out_features": num_classes}},
    ]
    return specs