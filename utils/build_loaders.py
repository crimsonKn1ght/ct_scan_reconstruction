import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import PairedLoDoPaBDataset


def build_loaders(
    root_dir,
    batch_size=6,
    num_workers=0,
    random_seed=42
):
    """
    Builds train, validation, and test loaders from explicitly labeled HDF5 pairs.
    """
    torch.manual_seed(random_seed)

    train_dataset = PairedLoDoPaBDataset(root_dir, split='train')
    val_dataset   = PairedLoDoPaBDataset(root_dir, split='validation')
    test_dataset  = PairedLoDoPaBDataset(root_dir, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader