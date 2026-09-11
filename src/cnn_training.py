from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import mat_mul
import modules.data_loaders as data_loader

import modules.convolution as conv

from datetime import datetime

import json


from modules.common import (
    trained_models_path, device,
    normalize_model_name, build_model,
    setup_seed, clean_gpu,
    dataset_name, train_loader, test_loader, _classes,
)
import numpy as np
import sys
import mqbench

def _debug_print(message: str):
    print(f"[DEBUG] {message}", flush=True)


def _describe_multiplier_input(multiplier_matrix: str | list[str]):
    if isinstance(multiplier_matrix, (list)):
        return f"{len(multiplier_matrix)} files" + f" ({', '.join(os.path.basename(m) for m in multiplier_matrix)})"
    if multiplier_matrix is None:
        return "None"
    return os.path.basename(multiplier_matrix)

def calibration(model, stats=False):
    """Calibrates model activations/weights using the training set."""
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
            if i >= 2048 // batch_size:
                break
            inputs = inputs.to(device)
            model(inputs)

    if not stats:
        for m in model.modules():
            if isinstance(m, conv.Conv2d_custom):
                m.freeze_qparams()


def set_data_loaders(model_name: str, cli_dataset_name: str = None):
    """Sets appropriate batch sizes based on the model architecture and loads data."""
    global train_loader, test_loader, _classes, batch_size, dataset_name

    name = model_name.lower()

    if cli_dataset_name is not None:
        dataset_name = cli_dataset_name.lower()
    else:
        if name == "lenet5":
            dataset_name = "mnist"
        elif name in ("vgg16", "alexnet_cifar10", "resnet8", "resnet20", "resnet"):
            dataset_name = "cifar10"
        elif name == "resnet56":
            dataset_name = "cifar100"
        else:
            raise ValueError(f"Cannot get dataset for model '{model_name}'. Specify --dataset explicitly or check the model name.")


    if name in ("lenet5", "resnet", "resnet8"):
        batch_size = 64
    elif name in ("vgg16", "alexnet_cifar10", "resnet56"):
        batch_size = 128

        

    train_loader, test_loader, _classes = data_loader.get_datasets(batch_size, dataset_name)


def get_exact_training_setup(model_name: str, model: nn.Module):
    """Returns specific hyperparameter configurations (epochs, optimizer, scheduler) per model."""
    name = model_name.lower()

    if name in ("resnet", "resnet8"):
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "lenet5":
        epochs = 20
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return epochs, optimizer, scheduler

    if name == "vgg16":
        epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "alexnet_cifar10":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    if name == "resnet56":
        epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return epochs, optimizer, scheduler

    # Default fallback setup
    epochs = 100
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return epochs, optimizer, scheduler


def train_one_epoch(epoch, model, optimizer, criterion):
    """Runs a single epoch of training and logs loss/accuracy."""
    print(f"Training epoch {epoch + 1}...")
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{batch:>5d}/{len(train_loader):>5d}]")
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {100.*correct/total:.2f}%")
    return avg_loss


def test(model):
    """Evaluates the model on the test dataset."""
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
        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
    return acc


def new_training_method(model_name: str, multiplier_matrix: str | list[str] = None, conv_type: int = 1,
                        bit_width: int = 8, signed: bool = False, zone: bool = False,
                        exact_accuracy: float = 0, no_retraining: bool = False):
    """Main pipeline handling full-precision, quantized, and approximate hardware simulation training."""
    
    input_name = _describe_multiplier_input(multiplier_matrix)
    _debug_print(f"new_training_method(model_name={model_name}, conv_type={conv_type}, bit_width={bit_width}, input={input_name}, multiple_layers={isinstance(multiplier_matrix, list)})")
    
    print(f"Network training with parameters: model_name={model_name}, conv_type={conv_type}, "
          f"bit_width={bit_width}, signed={signed}, input={input_name}, dataset={dataset_name}")

    models_dir = trained_models_path.rstrip('/')
    os.makedirs(models_dir, exist_ok=True)

    # Define paths for checkpoints
    exact_path = os.path.join(models_dir, f"{model_name}_{dataset_name}.pth")
    quant_path = os.path.join(models_dir, f"{model_name}_{dataset_name}_q{bit_width}.pth")

    #approx_tag = os.path.splitext(input_name)[0] if multiplier_matrix is isinstance(multiplier_matrix, str) else "default"
    if isinstance(multiplier_matrix, str):
        mult_matrix_specs = [f"{os.path.basename(multiplier_matrix)}"]
    elif isinstance(multiplier_matrix, list):
        mult_matrix_specs = [f"{os.path.basename(m)}" for m in multiplier_matrix]
    else:
        mult_matrix_specs = ["m=default"]
        
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  
    approx_noretrain_path = os.path.join(
        models_dir, f"{model_name}_{dataset_name}_{timestamp}_noretrain.pth"
    )
    approx_retrained_best_path = os.path.join(
        models_dir, f"{model_name}_{dataset_name}_{timestamp}_retrained_best.pth"
    )

    config_path = os.path.join(models_dir, "config", f"{timestamp}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config_specs = {
        "model_name": model_name,
        "conv_type": conv_type,
        "bit_width": bit_width,
        "signed": signed,
        "zone": zone,
        "dataset_name": dataset_name,
        "multiplier_matrix": mult_matrix_specs,
    }


    num_classes = _classes if _classes else 10

    if (model_name.lower() != "resnet8" and model_name.lower() != "resnet") and isinstance(multiplier_matrix, (list, tuple)):
        multiplier_matrix = multiplier_matrix[0] if multiplier_matrix else None

    # ---- conv_type 1: Exact (FP32) Model ----
    if conv_type == 1:
        model = build_model(model_name, conv_type=1, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        if os.path.exists(exact_path):
            print("Loading exact model and starting evaluation...")
            model.load_state_dict(torch.load(exact_path, weights_only=True))
            return test(model)
            
        print("Training exact model from scratch...")
        epochs, optimizer, scheduler = get_exact_training_setup(model_name, model)
        criterion = nn.CrossEntropyLoss()
        final_loss = None
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            final_loss = train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
        torch.save(model.state_dict(), exact_path)
        
        if final_loss is not None:
            print(f"FINAL_LOSS: {final_loss:.6f}")

        return test(model)

    # ---- conv_type 2: Quantized Model (QAT) ----
    if conv_type == 2:
        exact_exists = os.path.exists(exact_path)
        quant_exists = os.path.exists(quant_path)
        if not exact_exists:
            raise RuntimeError("Please train the exact model first.")
            
        model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        if not quant_exists:
            print("Starting quantized fine-tuning (5 epochs)...")
            model.load_state_dict(torch.load(exact_path, weights_only=True), strict=False)
            calibration(model)
            criterion = nn.CrossEntropyLoss()
            lr = 0.001 if bit_width == 4 else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
            best_acc = 0.0
            final_loss = None
            for epoch in range(5):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                final_loss = train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
                acc = test(model)
                best_acc = max(best_acc, acc)
            torch.save(model.state_dict(), quant_path)

            if final_loss is not None:
                print(f"FINAL_LOSS: {final_loss:.6f}")

            return best_acc
            
        print("Evaluating pre-existing quantized model...")
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        return test(model)

    # ---- conv_type 3: Approximate Computing Model ----
    if conv_type == 3:
        if not os.path.exists(quant_path):
            raise RuntimeError("Please train the quantized model first.")
            
        print("Retraining approximate model (3 epochs)...")
        model = build_model(model_name, conv_type=3, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        
        if no_retraining:
            acc = test(model)
            torch.save(model.state_dict(), approx_noretrain_path)
            print(f"Saved approximate (no-retrain) checkpoint to: {approx_noretrain_path}")

            with open(config_path, "w") as f:
                json.dump(config_specs, f, indent=4)
            print(f"Saved training configuration to: {config_path}")

            return acc
            
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if bit_width == 4 else 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        best_accuracy = 0
        best_state = None

        final_loss = None
        
        for epoch in range(3):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            final_loss = train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
            acc = test(model)
            # Early stop if accuracy drops drastically below baseline
            if acc < exact_accuracy - 3:
                print("Accuracy drop too high: not_good_enough")
                return acc
            if acc > best_accuracy:
                best_accuracy = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if final_loss is not None:
            print(f"FINAL_LOSS: {final_loss:.6f}")

        checkpoint = best_state if best_state is not None else model.state_dict()
        torch.save(checkpoint, approx_retrained_best_path)
        print(f"Saved approximate (retrained-best) checkpoint to: {approx_retrained_best_path}")

        with open(config_path, "w") as f:
            json.dump(config_specs, f, indent=4)
        print(f"Saved training configuration to: {config_path}")

        return best_accuracy

    # ---- conv_type 5: Calibration Statistics Collection ----
    if conv_type == 5:
        model = build_model(model_name, conv_type=5, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        calibration(model, True)
        print("Calibration for stats Done")
        return None

    raise ValueError(f"conv_type={conv_type} is not supported.")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training/evaluation of exact, quantized, and approximate CNNs.")
    parser.add_argument("--model_name", type=str, default="resnet")
    parser.add_argument("--conv_type", type=int, default=1)
    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--signed", action="store_true", default=False)
    parser.add_argument("--zone", action="store_true", default=False)
    parser.add_argument("--input_path", nargs="?", default=None)
    parser.add_argument("--exact_accuracy", type=float, default=0)
    parser.add_argument("--no_retraining", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "mnist"], required=False, help="Which dataset to use for training and evaluation.")

    parser.add_argument("--multiple_layers", action="store_true", default=False, help="if set use one multiplier matrix for each layer, otherwise use the same for all layers, input can be a single .npy corresponding to a single layer or a folder with multiple .npy that can correspond all to a single layer or each to a different layer based on --layer_mode (filemanes must contain layer)")
    parser.add_argument("--layer_mode", choices=["1", "2"], default="2", help="To use when --multiple_layers is set, if 1 it does one training per file; and use each file in the folder for the same corresponding layer, if 2 do a single training and use each file in the folder for the corresponding layer, if a layer has no corresponding file it will use the default multiplier")

    args = parser.parse_args()

    model_name = normalize_model_name(args.model_name)
    start = time.time()
    p = args.input_path


    _debug_print(f"CLI args: multiple_layers={args.multiple_layers}, layer_mode={args.layer_mode}, input_path={p}")

    # Scenario 1: Using the same approximate multiplier for all layers
    if not args.multiple_layers:
        # Scenario 1.1: No input path provided -> run exact pipeline only
        if p is None:
            setup_seed(int(args.seed))
            set_data_loaders(model_name, args.dataset)
            acc = new_training_method(model_name, None, args.conv_type, args.bit_width,
                                    args.signed, args.zone, args.exact_accuracy)
            print(f"Exact model accuracy: {acc}")
            sys.exit(0)

        if not os.path.exists(p):
            print(f"Error: The input path '{p}' does not exist.")
            sys.exit(1)

        # Scenario 1.2: Input path is a single multiplier matrix file
        if os.path.isfile(p):
            setup_seed(int(args.seed))
            set_data_loaders(model_name, args.dataset)
            acc = new_training_method(model_name, p, args.conv_type, args.bit_width,
                                    args.signed, args.zone, args.exact_accuracy, args.no_retraining)
            print(f"FINAL_ACCURACY:{acc}")
            clean_gpu()
            sys.exit(0)

        # Scenario 1.3: Input path is a directory -> batch evaluate all .npy files
        results = {}
        for f in os.listdir(p):
            if not f.endswith(".npy"):
                continue
            file_path = os.path.join(p, f)
            setup_seed(int(args.seed))
            set_data_loaders(model_name, args.dataset)
            acc = new_training_method(model_name, file_path, args.conv_type, args.bit_width,
                                    args.signed, args.zone, args.exact_accuracy, args.no_retraining)
            print(f"FINAL_ACCURACY:{acc}")
            results[f] = acc
            clean_gpu()

        print("Batch results dictionary:", results)
        print(f"Total training time: {time.time() - start:.2f} seconds")

    # Scenario 2: Using different approximate multipliers for each layer
    else:
        # Scenario 2.1: Layer mode 1 - Use a single npy file for the corresponding layer, iterate over
        #  all files for a single layer in a directory, layers with no corresponding npy file it will use the default multiplier    
        if args.layer_mode == "1":
            if p is None or not os.path.exists(p):
                print(f"Error: The input path '{p}' does not exist or is not a directory.")
                sys.exit(1)

            setup_seed(int(args.seed))
            set_data_loaders(model_name, args.dataset)

            if not os.path.isdir(p):
                file_list = [p] if p.endswith(".npy") else []
            else:   
                file_list = [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".npy")]
                if not file_list:
                    print(f"Error: No .npy files found in the directory '{p}'.")
                    sys.exit(1)
            _debug_print("layer_mode=1 file_list=" + ", ".join(os.path.basename(f) for f in file_list))
            results = {}
            for file in file_list:
                acc = new_training_method(
                    model_name,
                    [file],
                    args.conv_type,
                    args.bit_width,
                    args.signed,
                    args.zone,
                    args.exact_accuracy,
                    args.no_retraining
                )
                print(f"FINAL_ACCURACY:{acc}")
                results[file] = acc
                clean_gpu()

            print("Batch results dictionary:", results)
            print(f"Total training time: {time.time() - start:.2f} seconds")

        # Scenario 2.2: Layer mode 2 - Use x npy files for the correspoding x layers, if a layer has no corresponding npy file it will use the default multiplier
        if args.layer_mode == "2":
            if p is None or not os.path.exists(p):
                print(f"Error: The input path '{p}' does not exist or is not a directory.")
                sys.exit(1)

            setup_seed(int(args.seed))
            set_data_loaders(model_name, args.dataset)

            if not os.path.isdir(p):
                file_list = [p] if p.endswith(".npy") else []
            else:   
                file_list = [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".npy")]
                if not file_list:
                    print(f"Error: No .npy files found in the directory '{p}'.")
                    sys.exit(1)
            _debug_print("layer_mode=2 file_list=" + ", ".join(os.path.basename(f) for f in file_list))
            
            acc = new_training_method(
                model_name,
                file_list,
                args.conv_type,
                args.bit_width,
                args.signed,
                args.zone,
                args.exact_accuracy,
                args.no_retraining
            )
            print(f"FINAL_ACCURACY:{acc}")
            clean_gpu()

    print(f"Total training time: {time.time() - start}")
