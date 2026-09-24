from __future__ import annotations

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "mnist": ((0.1307,), (0.3081,)),
}


def _get_transforms(dataset_key: str, image_size: int):
    """Restituisce le trasformazioni di Training e Testing/Validation in base alla dimensione dell'immagine."""
    mean, std = MEAN_STD[dataset_key]

    if image_size == 32:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # Es. 256 per 224x224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_tf, test_tf


def get_cifar10(batch_size: int, data_root: str = "./data/cifar10", image_size: int = 32, **kwargs):
    num_workers = kwargs.setdefault("num_workers", 2)
    pin_memory = kwargs.setdefault("pin_memory", True)
    print("Building CIFAR 10")

    kwargs.pop("input_size", None)

    train_tf, test_tf = _get_transforms("cifar10", image_size)
    eval_no_aug_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)) if image_size != 32 else (lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD["cifar10"]),
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    train_dataset_no_aug = datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_no_aug_tf)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    train_loader_no_aug = torch.utils.data.DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, train_loader_no_aug, len(train_dataset.classes)


def get_cifar100(batch_size: int, data_root: str = "./data/cifar100", image_size: int = 32, **kwargs):
    num_workers = kwargs.setdefault("num_workers", 2)
    pin_memory = kwargs.setdefault("pin_memory", True)
    print("Building CIFAR 100")
    kwargs.pop("input_size", None)

    train_tf, test_tf = _get_transforms("cifar100", image_size)
    eval_no_aug_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)) if image_size != 32 else (lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD["cifar100"]),
    ])

    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tf)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)
    train_dataset_no_aug = datasets.CIFAR100(root=data_root, train=True, download=True, transform=eval_no_aug_tf)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    train_loader_no_aug = torch.utils.data.DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader, train_loader_no_aug, len(train_dataset.classes)


def get_imagenet(batch_size: int, data_root: str = "./data/tiny-imagenet-200", image_size: int = 64, **kwargs):
    """Data loader per Tiny ImageNet a risoluzione corretta (64x64 default) e classi allineate."""
    num_workers = kwargs.setdefault("num_workers", 4)
    pin_memory = kwargs.setdefault("pin_memory", True)
    print("Building imagenet (Tiny ImageNet mode)")
    
    train_tf, test_tf = _get_transforms("imagenet", image_size)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    test_dataset = datasets.ImageFolder(val_dir, transform=test_tf)

    test_dataset.class_to_idx = train_dataset.class_to_idx
    test_dataset.classes = train_dataset.classes

    aligned_samples = []
    for path, _ in test_dataset.samples:
        folder_name = os.path.basename(os.path.dirname(path))
        if folder_name in train_dataset.class_to_idx:
            aligned_samples.append((path, train_dataset.class_to_idx[folder_name]))
            
    test_dataset.samples = aligned_samples
    test_dataset.targets = [s[1] for s in aligned_samples]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=g, worker_init_fn=seed_worker, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, **kwargs
    )

    return train_loader, test_loader, len(train_dataset.classes)

def get_mnist(batch_size: int, data_root: str = "./data/mnist", image_size: int = 32, **kwargs):
    num_workers = kwargs.setdefault("num_workers", 1)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD["mnist"])
    ])

    train_dataset = datasets.MNIST(root=data_root, train=True, transform=tf, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=tf, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, len(train_dataset.classes)


def get_datasets(batch_size: int, dataset_name: str, image_size: int = None, **kwargs):
    """
    Funzione d'ingresso unificata.
    Se image_size non viene specificato, adotta la risoluzione standard del dataset.
    """
    if not dataset_name:
        raise ValueError("dataset_name non può essere None o vuoto.")

    dataset = dataset_name.lower().strip()

    if dataset == "mnist":
        img_sz = image_size or 32
        train_loader, test_loader, num_classes = get_mnist(batch_size=batch_size, image_size=img_sz, **kwargs)
        return train_loader, test_loader, num_classes

    elif dataset in ["cifar10", "cifar-10"]:
        img_sz = image_size or 32
        train_loader, test_loader, _, num_classes = get_cifar10(batch_size=batch_size, image_size=img_sz, **kwargs)
        return train_loader, test_loader, num_classes

    elif dataset in ["cifar100", "cifar-100"]:
        img_sz = image_size or 32
        train_loader, test_loader, _, num_classes = get_cifar100(batch_size=batch_size, image_size=img_sz, **kwargs)
        return train_loader, test_loader, num_classes

    elif dataset in ["imagenet", "imagenet1k", "imagenet-1k"]:
        img_sz = image_size or 224
        train_loader, test_loader, num_classes = get_imagenet(batch_size=batch_size, image_size=img_sz, **kwargs)
        return train_loader, test_loader, num_classes

    else:
        raise ValueError(f"Dataset non supportato: '{dataset_name}'")
