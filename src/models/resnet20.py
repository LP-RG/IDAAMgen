import torch
import torch.nn as nn
import modules.convolution as cc  
import os

def _debug_print(message: str):
    print(f"[DEBUG] {message}", flush=True)


def _normalize_multiplier_matrix(multiplier_matrix: str | list[str]) -> str | dict[str, str] | None:
    if not isinstance(multiplier_matrix, (list)):
        return multiplier_matrix

    keys = {'s', '1a_1', '1a_2', '1b_1', '1b_2', '1c_1', '1c_2', '2a_1', '2a_2', '2a_s', '2b_1', '2b_2', '2c_1', '2c_2', '3a_1', '3a_2', '3a_s', '3b_1', '3b_2', '3c_1', '3c_2'}
    normalized_multiplier_matrix: dict[str, str] = {}

    for multiplier_matrix_file in multiplier_matrix:
        matrix_name = os.path.splitext(os.path.basename(multiplier_matrix_file))[0]
        for key in keys:
            if f"_{key}" in matrix_name:
                normalized_multiplier_matrix[key] = multiplier_matrix_file
                break
    print(f"[DEBUG] normalized multiplier mapping: {normalized_multiplier_matrix}", flush=True)
    if not normalized_multiplier_matrix:
        return None

    _debug_print(
        "normalized multiplier mapping: "
        + ", ".join(f"{key}={os.path.basename(value)}" for key, value in sorted(normalized_multiplier_matrix.items()))
    )

    return normalized_multiplier_matrix


def _resolve_multiplier_matrix(multiplier_matrix: str | dict[str, str] | None, key: str) -> str | None:
    if isinstance(multiplier_matrix, dict):
        if key in multiplier_matrix:
            return multiplier_matrix[key]

        return None
    return multiplier_matrix


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    """BasicBlock stile ResNet-18/34 (expansion=1) con Conv2d_custom."""
    expansion = 1
    def __init__(
        self,
        in_channels,
        out_channels,
        multiplier_matrix: str | list[str],
        stride=1,
        conv_type=1,
        bit_width=8,
        signed=False,
        name="0",
    ):
        super().__init__()
        multiplier_matrix = _normalize_multiplier_matrix(multiplier_matrix)
        self.conv1 = cc.Conv2d_custom(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name=name + "_1", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_1"),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = cc.Conv2d_custom(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name=name + "_2", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_2"),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Option B: 1x1 conv per adattare dimensioni/canali nello shortcut
            self.shortcut = nn.Sequential(
                cc.Conv2d_custom(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0, bias=False,
                    conv_type=conv_type, bit_width=bit_width, signed=signed,
                    name=name + "_s", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_s"),
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet20(nn.Module):
    """
    ResNet-20 per CIFAR (6n+2 con n=3): canali 16-32-64, tre blocchi per stage.
    Mantiene firma e flag "custom" come nel tuo esempio.
    """
    def __init__(
        self,
        multiplier_matrix: str | list[str],
        num_classes=10,
        conv_type=1,
        bit_width=8,
        signed=False,
        zone=False,
    ):
        super().__init__()
        multiplier_matrix = _normalize_multiplier_matrix(multiplier_matrix)
        _debug_print(f"ResNet20 init conv_type={conv_type}, bit_width={bit_width}, zone={zone}, multiplier_matrix_type={type(multiplier_matrix).__name__}")
        
        self.conv1 = cc.Conv2d_custom(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
            name="s", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, "s"))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Tre stage con 3 blocchi ciascuno (n=3)
        self.layer1 = self._make_layer(
            block=BasicBlock, in_channels=16, out_channels=16, blocks=3, stride=1,
            base_name="1", multiplier_matrix=multiplier_matrix,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
        )
        self.layer2 = self._make_layer(
            block=BasicBlock, in_channels=16, out_channels=32, blocks=3, stride=2,
            base_name="2", multiplier_matrix=multiplier_matrix,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
        )
        self.layer3 = self._make_layer(
            block=BasicBlock, in_channels=32, out_channels=64, blocks=3, stride=2,
            base_name="3", multiplier_matrix=multiplier_matrix,
            conv_type=conv_type, bit_width=bit_width, signed=signed,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        # Inizializzazione standard per ResNet
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(
        self, block, in_channels, out_channels, blocks, stride, base_name,
        multiplier_matrix, conv_type, bit_width, signed,
    ):
        layers = []
        # primo blocco può fare downsample
        layers.append(block(
            in_channels=in_channels,
            out_channels=out_channels,
            multiplier_matrix=multiplier_matrix,
            stride=stride,
            conv_type=conv_type,
            bit_width=bit_width,
            signed=signed,
            name=f"{base_name}a",
        ))
        # blocchi rimanenti a stride 1
        for i in range(1, blocks):
            suffix = chr(ord('a') + i)  # 'b', 'c', ...
            layers.append(block(
                in_channels=out_channels * block.expansion,
                out_channels=out_channels,
                multiplier_matrix=multiplier_matrix,
                stride=1,
                conv_type=conv_type,
                bit_width=bit_width,
                signed=signed,
                name=f"{base_name}{suffix}",
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)   # 16
        out = self.layer2(out)   # 32
        out = self.layer3(out)   # 64
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
