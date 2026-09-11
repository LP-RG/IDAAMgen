from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn as nn
import modules.convolution as cc

def _debug_print(message: str):
    print(f"[DEBUG] {message}", flush=True)


def _normalize_multiplier_matrix(multiplier_matrix: str | list[str]) -> str | dict[str, str] | None:
    if not isinstance(multiplier_matrix, (list)):
        return multiplier_matrix

    keys = {"1_1", "1_2", "2_1", "2_2", "2_s", "3_1", "3_2", "3_s", "s_8"}
    normalized_multiplier_matrix: dict[str, str] = {}

    for multiplier_matrix_file in multiplier_matrix:
        matrix_name = os.path.splitext(os.path.basename(multiplier_matrix_file))[0]
        for key in keys:
            if f"_{key}" in matrix_name:
                normalized_multiplier_matrix[key] = multiplier_matrix_file
                break

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
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, features):
        return self.lambd(features)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier_matrix: str | list[str], stride=1, conv_type = 1, bit_width = 8, signed =  False, name = "0"):
        super(BasicBlock, self).__init__()

        multiplier_matrix = _normalize_multiplier_matrix(multiplier_matrix)

        self.conv1 = cc.Conv2d_custom(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                    conv_type=conv_type, bit_width=bit_width, signed= signed, name = name + "_1", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_1"))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = cc.Conv2d_custom(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                    conv_type=conv_type, bit_width=bit_width, signed= signed, name = name + "_2", multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_2"))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_multiplier_matrix = _resolve_multiplier_matrix(multiplier_matrix, name + "_s")
            self.shortcut = nn.Sequential(
                cc.Conv2d_custom(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, 
                    conv_type=conv_type, bit_width=bit_width, signed= signed, name = name + "_s", multiplier_matrix=shortcut_multiplier_matrix
                ),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, multiplier_matrix: str | list[str], num_classes=10, conv_type = 1, bit_width = 8, signed = False, zone = False):
        super(ResNet8, self).__init__()

        multiplier_matrix = _normalize_multiplier_matrix(multiplier_matrix)
        _debug_print(f"ResNet8 init conv_type={conv_type}, bit_width={bit_width}, zone={zone}, multiplier_matrix_type={type(multiplier_matrix).__name__}")

        #Keeping first layer unapproximated
        first_layer_conv_type = conv_type if (conv_type == 1 or conv_type == 5) else 2
        self.conv1 = cc.Conv2d_custom(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                    conv_type=first_layer_conv_type if zone else conv_type, bit_width=bit_width, signed= signed, name = "s_8", multiplier_matrix=_resolve_multiplier_matrix(multiplier_matrix, "s_8"))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = BasicBlock(16, 16, stride=1, conv_type=conv_type, bit_width=bit_width, signed=signed, name = "1", multiplier_matrix=multiplier_matrix)
        self.layer2 = BasicBlock(16, 32, stride=2, conv_type=conv_type, bit_width=bit_width, signed=signed, name = "2", multiplier_matrix=multiplier_matrix)
        self.layer3 = BasicBlock(32, 64, stride=2, conv_type=conv_type, bit_width=bit_width, signed=signed, name = "3", multiplier_matrix=multiplier_matrix)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
