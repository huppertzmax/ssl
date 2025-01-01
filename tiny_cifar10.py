from pytorch_lightning import Trainer
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import TrialCIFAR10
from torchvision import transforms as transform_lib
from torch.utils.data import DataLoader, Subset
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import numpy as np
import torch

class TinyCIFAR10DataModule(VisionDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset",
        num_samples: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: int = 0,
        seed: int = 42,
    ):
        super().__init__(data_dir, batch_size=batch_size, num_workers=num_workers)
        self.num_samples = num_samples
        self.val_split = val_split
        self.seed = seed
        self.transform = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])


        self.train_dataset = TrialCIFAR10(self.data_dir, train=True, download=True, num_samples=self.num_samples, labels=list(range(10)))
        #indices = self._get_balanced_subset_indices(full_dataset)
        #subset = Subset(full_dataset, indices)
        #val_size = self.val_split

        #self.train_dataset, self.val_dataset = self._split_dataset(subset, val_size)
        self.test_dataset = TrialCIFAR10(self.data_dir, train=False, download=True, num_samples=self.num_samples//2, labels=list(range(10)))

        print(torch.bincount(self.train_dataset.targets))
        print(torch.bincount(self.test_dataset.targets))
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def _get_balanced_subset_indices(self, dataset):
        labels = np.array(dataset.targets)
        indices = []

        for label in range(10): 
            label_indices = np.where(labels == label)[0]
            selected_indices = np.random.choice(label_indices, self.num_samples, replace=False)
            indices.extend(selected_indices)

        return indices

    def _split_dataset(self, dataset, val_size):
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return train_subset, val_subset

