import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import time
import mat_mul
import modules.data_loaders as data_loader

from modules.common import (
    trained_models_path, device,
    normalize_model_name, build_model,
    setup_seed, clean_gpu,
    train_loader, test_loader, _classes,
)


def calibration(model, stats=False):
    """Calibrates model activations/weights using the training set."""
    print("Calibrating model...")
    if stats:
        model.eval()
    else:
        model.train()
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        model(inputs)

def set_data_loaders(model_name: str, batch_size: int = 64):
    """Sets appropriate batch sizes based on the model architecture and loads data."""
    global train_loader, test_loader, _classes
    name = model_name.lower()

    if name in ("lenet5", "resnet", "resnet8"):
        batch_size = 64
    elif name in ("vgg16", "alexnet_cifar10", "resnet56"):
        batch_size = 128

    train_loader, test_loader, _classes = data_loader.get_datasets(batch_size, model_name)


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
    print(f"Epoch {epoch + 1}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")


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


def new_training_method(model_name: str, multiplier_matrix=None, conv_type: int = 1,
                        bit_width: int = 8, signed: bool = False, zone: bool = False,
                        exact_accuracy: float = 0, no_retraining: bool = False, shift_bits = 0):
    """Main pipeline handling full-precision, quantized, and approximate hardware simulation training."""
    input_name = multiplier_matrix.split("/")[-1] if multiplier_matrix is not None else "None"
    print(f"Network training with parameters: model_name={model_name}, conv_type={conv_type}, "
          f"bit_width={bit_width}, signed={signed}, input={input_name}")

    models_dir = trained_models_path.rstrip('/')
    os.makedirs(models_dir, exist_ok=True)

    # Define paths for checkpoints
    exact_path = os.path.join(models_dir, f"{model_name}.pth")
    quant_path = os.path.join(models_dir, f"{model_name}_q{bit_width}.pth")

    approx_tag = os.path.splitext(input_name)[0] if multiplier_matrix is not None else "default"
    approx_noretrain_path = os.path.join(
        models_dir, f"{model_name}_a{bit_width}_{approx_tag}_noretrain.pth"
    )
    approx_retrained_best_path = os.path.join(
        models_dir, f"{model_name}_a{bit_width}_{approx_tag}_retrained_best.pth"
    )

    num_classes = _classes if _classes else 10

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
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
        torch.save(model.state_dict(), exact_path)
        return test(model)

    # ---- conv_type 2: Quantized Model (QAT) ----
    if conv_type == 2 and shift_bits == 0:
        exact_exists = os.path.exists(exact_path)
        quant_exists = os.path.exists(quant_path)
        if not exact_exists:
            raise RuntimeError("Please train the exact model first.")
            
        model = build_model(model_name, conv_type=2, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes, shift_bits=shift_bits)
        if not quant_exists:
            print("Starting quantized fine-tuning (5 epochs)...")
            model.load_state_dict(torch.load(exact_path, weights_only=True), strict=False)
            calibration(model)
            criterion = nn.CrossEntropyLoss()
            lr = 0.001 if bit_width == 4 else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
            best_acc = 0.0
            for epoch in range(5):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                train_one_epoch(epoch, model, optimizer, criterion)
                scheduler.step()
                acc = test(model)
                best_acc = max(best_acc, acc)
            torch.save(model.state_dict(), quant_path)
            return best_acc
            
        print("Evaluating pre-existing quantized model...")
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        return test(model)

    # ---- conv_type 3: Approximate Computing Model ----
    if conv_type == 3 or shift_bits != 0:
        if not os.path.exists(quant_path):
            raise RuntimeError("Please train the quantized model first.")
            
        print("Retraining approximate model (3 epochs)...")
        model = build_model(model_name, conv_type=conv_type, bit_width=bit_width, signed=signed,
                            zone=zone, multiplier_matrix=multiplier_matrix, num_classes=num_classes , shift_bits=shift_bits)
        model.load_state_dict(torch.load(quant_path, weights_only=True))
        calibration(model)
        
        if no_retraining:
            acc = test(model)
            torch.save(model.state_dict(), approx_noretrain_path)
            print(f"Saved approximate (no-retrain) checkpoint to: {approx_noretrain_path}")
            return acc
            
        criterion = nn.CrossEntropyLoss()
        lr = 0.001 if bit_width == 4 else 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        best_accuracy = 0
        best_state = None
        
        for epoch in range(3):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_one_epoch(epoch, model, optimizer, criterion)
            scheduler.step()
            acc = test(model)
            # Early stop if accuracy drops drastically below baseline
            if acc < exact_accuracy - 3:
                print("Accuracy drop too high: not_good_enough")
                return acc
            if acc > best_accuracy:
                best_accuracy = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                
        checkpoint = best_state if best_state is not None else model.state_dict()
        torch.save(checkpoint, approx_retrained_best_path)
        print(f"Saved approximate (retrained-best) checkpoint to: {approx_retrained_best_path}")
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
    parser.add_argument("--seed",default=42,required=False)
    parser.add_argument("--shift_bits", type=int, default=0)
    args = parser.parse_args()

    model_name = normalize_model_name(args.model_name)
    start = time.time()
    p = args.input_path

    # Scenario 1: No input path provided -> run exact pipeline only
    if p is None:
        setup_seed(args.seed)
        set_data_loaders(model_name)
        acc = new_training_method(model_name, None, args.conv_type, args.bit_width,
                                  args.signed, args.zone, args.exact_accuracy, shift_bits=args.shift_bits)
        print(f"Exact model accuracy: {acc}")
        sys.exit(0)

    if not os.path.exists(p):
        print(f"Error: The input path '{p}' does not exist.")
        sys.exit(1)

    # Scenario 2: Input path is a single multiplier matrix file
    if os.path.isfile(p):
        setup_seed(args.seed)
        set_data_loaders(model_name)
        acc = new_training_method(model_name, p, args.conv_type, args.bit_width,
                                  args.signed, args.zone, args.exact_accuracy, args.no_retraining,shift_bits= args.shift_bits)
        print(f"FINAL_ACCURACY:{acc}")
        clean_gpu()
        sys.exit(0)

    # Scenario 3: Input path is a directory -> batch evaluate all .npy files
    results = {}
    for f in os.listdir(p):
        if not f.endswith(".npy"):
            continue
        file_path = os.path.join(p, f)
        setup_seed(args.seed)
        set_data_loaders(model_name)
        acc = new_training_method(model_name, file_path, args.conv_type, args.bit_width,
                                  args.signed, args.zone, args.exact_accuracy, args.no_retraining, shift_bits=args.shift_bits)
        print(f"FINAL_ACCURACY:{acc}")
        results[f] = acc
        clean_gpu()

    print("Batch results dictionary:", results)
    print(f"Total training time: {time.time() - start:.2f} seconds")