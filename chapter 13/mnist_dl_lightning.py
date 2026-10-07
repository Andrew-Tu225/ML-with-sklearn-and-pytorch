import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split


class MNISTDataModule(pl.LightningDataModule):
    """
    MNIST DataModule for PyTorch Lightning.
    """
    def __init__(self, batch_size=64, data_path="./"):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def prepare_data(self):
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(self, stage=None):
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )
        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)