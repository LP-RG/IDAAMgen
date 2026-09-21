import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

def get_cifar10(batch_size, data_root, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print(f"Building CIFAR-10 data loader with {num_workers} workers")

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    train_loader_no_aug = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, train_loader_no_aug, num_classes


def get_cifar100(batch_size, data_root, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print(f"Building CIFAR-100 data loader with {num_workers} workers")

    train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    )

    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    train_loader_no_aug = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, train_loader_no_aug, num_classes


def get_mnist(batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,))
        ]),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, num_classes


def get_datasets(batch_size, dataset_name):

    if dataset_name is not None:
        dataset = dataset_name.lower()
        if dataset == "mnist":
            train_loader, test_loader, num_classes = get_mnist(batch_size=batch_size)
            return train_loader, test_loader, num_classes
        elif dataset == "cifar10":
            train_loader, test_loader, _, num_classes = get_cifar10(
                batch_size=batch_size, data_root='./data/cifar'
            )
            return train_loader, test_loader, num_classes
        elif dataset == "cifar100":
            train_loader, test_loader, _, num_classes = get_cifar100(
                batch_size=batch_size, data_root='./data/cifar100'
            )
            return train_loader, test_loader, num_classes
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
