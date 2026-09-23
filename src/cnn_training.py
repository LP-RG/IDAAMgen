from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import mat_mul
import modules.convolution as conv
import modules.data_loaders as data_loader
from modules.common import (
    ROOT_DIR,
    build_model,
    clean_gpu,
    device,
    normalize_model_name,
    setup_seed,
    trained_models_path,
)

train_loader = None
test_loader = None
_classes = None
batch_size = 64
dataset_name = None


def calibration(model: nn.Module, stats: bool = False):
    """Calibra attivazioni e pesi del modello usando il training set."""
    print("Calibrating model...")

    for m in model.modules():
        if isinstance(m, conv.Conv2d_custom):
            m.calibrating = not stats

    if stats:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= 1024 // batch_size:
                break
            inputs = inputs.to(device)
            model(inputs)

    if not stats:
        for m in model.modules():
            if isinstance(m, conv.Conv2d_custom):
                m.freeze_qparams()


def set_data_loaders(model_name: str, cli_dataset_name: str = None):
    """
    Seleziona automaticamente il dataset, la dimensione dell'immagine (32x32 vs 224x224) 
    e il batch size ottimale in base all'architettura SOTA selezionata.
    """
    global train_loader, test_loader, _classes, batch_size, dataset_name

    norm_name = normalize_model_name(model_name)

    if cli_dataset_name is not None:
        dataset_name = cli_dataset_name.lower()
    else:
        if norm_name == "resnet20_cifar":
            dataset_name = "cifar10"
        elif norm_name in ("resnext50_32x4d", "mobilenet_v2", "convnext_tiny", "vgg16_bn"):
            dataset_name = "imagenet"
        elif norm_name == "resnet50":
            if cli_dataset_name == "cifar100":
                dataset_name = "cifar100"
            else:
                dataset_name = "imagenet"

    image_size = 32 if "cifar" in dataset_name or dataset_name == "mnist" else 224

    if norm_name == "resnet20_cifar":
        batch_size = 128
    elif norm_name in ("mobilenet_v2", "vgg16_bn"):
        batch_size = 64
    elif norm_name in ("resnet50", "resnext50_32x4d", "convnext_tiny"):
        batch_size = 32
    else:
        batch_size = 64

    train_loader, test_loader, _classes = data_loader.get_datasets(
        batch_size=batch_size,
        dataset_name=dataset_name,
        image_size=image_size
    )


def get_exact_training_setup(model_name: str, model: nn.Module):
    norm_name = normalize_model_name(model_name)

    if norm_name == "resnet20_cifar":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    elif norm_name in ("resnet50", "resnext50_32x4d"):
        epochs = 100
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return epochs, optimizer, scheduler

    elif norm_name == "mobilenet_v2":
        epochs = 150
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=4e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return epochs, optimizer, scheduler

    elif norm_name == "convnext_tiny":
        epochs = 100
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return epochs, optimizer, scheduler

    elif norm_name == "vgg16_bn":
        epochs = 90
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return epochs, optimizer, scheduler

    epochs = 100
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return epochs, optimizer, scheduler


def train_one_epoch(epoch: int, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module):
    """Esegue un'epoca di addestramento e calcola loss/accuratezza."""
    print(f"Training epoch {epoch + 1}...")
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch % 100 == 0:
            print(f"Loss: {loss.item():>7f}  [{batch:>5d}/{len(train_loader):>5d}]")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Avg Loss: {avg_loss:.4f}, Accuracy: {100. * correct / total:.2f}%")
    return avg_loss


def test(model: nn.Module) -> float:
    """Valuta la rete sul test set."""
    print("Testing model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print("Targets:", targets[:500])
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def new_training_method(
    model_name: str,
    multiplier_matrix=None,
    conv_type: int = 1,
    bit_width: int = 8,
    signed: bool = False,
    zone: bool = False,
    exact_accuracy: float = 0,
    no_retraining: bool = False,
    shift_bits: int = 0,
):
    """Pipeline principale di training/finetuning (FP32, Quantizzato e Approssimato)."""
    input_name = os.path.basename(multiplier_matrix) if isinstance(multiplier_matrix, str) else "None"

    print(f"\n[EXECUTION] Model: {model_name} | ConvType: {conv_type} | BitWidth: {bit_width} | "
          f"Signed: {signed} | Multiplier: {input_name} | Dataset: {dataset_name}")

    models_dir = trained_models_path.rstrip('/')
    os.makedirs(models_dir, exist_ok=True)

    exact_path = os.path.join(models_dir, f"{model_name}_{dataset_name}.pth")
    quant_path = os.path.join(models_dir, f"{model_name}_{dataset_name}_q{bit_width}.pth")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    approx_noretrain_path = os.path.join(models_dir, f"{model_name}_{dataset_name}_{timestamp}_noretrain.pth")
    approx_retrained_best_path = os.path.join(models_dir, f"{model_name}_{dataset_name}_{timestamp}_retrained_best.pth")

    config_path = os.path.join(models_dir, "config", f"{timestamp}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config_specs = {
        "model_name": model_name,
        "conv_type": conv_type,
        "bit_width": bit_width,
        "signed": signed,
        "zone": zone,
        "dataset_name": dataset_name,
        "multiplier_matrix": input_name,
    }

    num_classes = _classes if _classes else 10

    if conv_type == 1:
        model = build_model(
            model_name, conv_type=1, bit_width=bit_width, signed=signed,
            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes,
            dataset_name=dataset_name,  
        )
        if os.path.exists(exact_path):
            print("Loading exact model checkpoint...")
            model.load_state_dict(torch.load(exact_path, map_location=device, weights_only=True))
            return test(model)

        print("Training exact model from scratch...")
        epochs, optimizer, scheduler = get_exact_training_setup(model_name, model)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()

        torch.save(model.state_dict(), exact_path)
        return test(model)

    if conv_type == 2 and shift_bits == 0:
        if not os.path.exists(exact_path):
            raise RuntimeError(f"Exact baseline not found at '{exact_path}'. Train conv_type=1 first.")

        model = build_model(
            model_name, conv_type=2, bit_width=bit_width, signed=signed,
            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes, shift_bits=shift_bits,
            dataset_name=dataset_name, 
        )

        if not os.path.exists(quant_path):
            print("Starting Quantization-Aware Fine-Tuning (5 epochs)...")
            model.load_state_dict(torch.load(exact_path, map_location=device, weights_only=True), strict=False)
            calibration(model)
            criterion = nn.CrossEntropyLoss()
            lr = 0.001 if bit_width == 4 else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

            best_acc = 0.0
            for epoch in range(5):
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
                acc = test(model)
                best_acc = max(best_acc, acc)

            torch.save(model.state_dict(), quant_path)
            return best_acc

        print("Loading pre-existing quantized model...")
        model.load_state_dict(torch.load(quant_path, map_location=device, weights_only=True))
        calibration(model)
        return test(model)

    if conv_type == 3 or shift_bits != 0:
        if not os.path.exists(quant_path):
            raise RuntimeError(f"Quantized baseline not found at '{quant_path}'. Train conv_type=2 first.")

        print("Instantiating approximate model with custom multiplier hardware matrix...")
        model = build_model(
            model_name, conv_type=conv_type, bit_width=bit_width, signed=signed,
            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes, shift_bits=shift_bits,
            dataset_name=dataset_name,  # 
        )
        model.load_state_dict(torch.load(quant_path, map_location=device, weights_only=True))
        calibration(model)

        if no_retraining:
            acc = test(model)
            torch.save(model.state_dict(), approx_noretrain_path)
            with open(config_path, "w") as f:
                json.dump(config_specs, f, indent=4)
            return acc

        print("Fine-tuning approximate hardware model (3 epochs)...")
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if bit_width == 4 else 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        best_accuracy = 0
        best_state = None

        for epoch in range(3):
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
            acc = test(model)

            if acc < exact_accuracy - 3.0:
                print("Accuracy drop too severe. Terminating early.")
                return acc

            if acc > best_accuracy:
                best_accuracy = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        checkpoint = best_state if best_state is not None else model.state_dict()
        torch.save(checkpoint, approx_retrained_best_path)
        with open(config_path, "w") as f:
            json.dump(config_specs, f, indent=4)

        return best_accuracy

    if conv_type == 5:
        model = build_model(
            model_name, conv_type=5, bit_width=bit_width, signed=signed,
            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes,
            dataset_name=dataset_name,
        )
        model.load_state_dict(torch.load(quant_path, map_location=device, weights_only=True))
        calibration(model)
        calibration(model, stats=True)
        print("Calibration statistics collection completed successfully.")
        return None

    raise ValueError(f"Unsupported conv_type: {conv_type}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training/evaluation CLI for SOTA DNNs with approximate computing.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet20_cifar", "resnext50_32x4d", "mobilenet_v2", "convnext_tiny", "vgg16_bn"],
        help="Target SOTA architecture name."
    )
    parser.add_argument("--conv_type", type=int, default=1, help="1=Exact, 2=Quantized, 3=Approximate, 5=Stats")
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--signed", action="store_true", default=False)
    parser.add_argument("--zone", action="store_true", default=False)
    parser.add_argument("--input_path", nargs="?", default=None, help="Path to matrix .npy file or directory of matrices.")
    parser.add_argument("--exact_accuracy", type=float, default=0)
    parser.add_argument("--no_retraining", action="store_true", default=False)
    parser.add_argument("--shift_bits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "imagenet", "mnist"],
        default="cifar100",
        help="Force a specific dataset (optional)."
    )

    args = parser.parse_args()

    model_name = normalize_model_name(args.model_name)
    start_time = time.time()
    p = args.input_path

    if p is None:
        setup_seed(args.seed)
        set_data_loaders(model_name, args.dataset)
        acc = new_training_method(
            model_name, None, args.conv_type, args.bit_width,
            args.signed, args.zone, args.exact_accuracy, shift_bits=args.shift_bits
        )
        print(f"\nFinal Accuracy: {acc:.2f}%")
        sys.exit(0)

    if not os.path.exists(p):
        print(f"Error: Path '{p}' does not exist.")
        sys.exit(1)

    if os.path.isfile(p):
        setup_seed(args.seed)
        set_data_loaders(model_name, args.dataset)
        acc = new_training_method(
            model_name, p, args.conv_type, args.bit_width,
            args.signed, args.zone, args.exact_accuracy, args.no_retraining, shift_bits=args.shift_bits
        )
        print(f"\nFINAL_ACCURACY: {acc:.2f}%")
        clean_gpu()
        sys.exit(0)

    results = {}
    for f in os.listdir(p):
        if not f.endswith(".npy"):
            continue
        file_path = os.path.join(p, f)
        setup_seed(args.seed)
        set_data_loaders(model_name, args.dataset)
        acc = new_training_method(
            model_name, file_path, args.conv_type, args.bit_width,
            args.signed, args.zone, args.exact_accuracy, args.no_retraining, shift_bits=args.shift_bits
        )
        print(f"File {f} -> FINAL_ACCURACY: {acc:.2f}%")
        results[f] = acc
        clean_gpu()

    print("\nBatch Evaluation Results Summary:", results)
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")