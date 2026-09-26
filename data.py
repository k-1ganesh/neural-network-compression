from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


DATA_ROOT = "data"
BATCH_SIZE = 256
NUM_WORKERS = 2


def get_dataloaders():
    """
    Train and Test dataloaders for Fashion MNIST dataset.
    Train : 60,000 images  | Test: 10,000 images
    Each image dimension : 28 * 28
    """
    
    # Pipeline of transformations.
    transform = transforms.Compose([
        
        # Converts the image to a PyTorch tensor. (28 * 28) -> (1 * 28 * 28)  Range [0, 255] -> [0, 1]
        transforms.ToTensor(),

        # Normalizes the tensor such that mean = 0 and std = 1.  For each pixel x :  (x - mean) / std.
        # Note: The mean and std are computed from the training set only and are applied to the test set as well.
        transforms.Normalize(
            mean=(0.2860,),
            std=(0.3530,)
        ),
    ])

    train_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
    )

    return train_loader, test_loader