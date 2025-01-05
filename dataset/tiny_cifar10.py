import torch
import numpy as np
import pytorch_lightning as pl
from pl_bolts.datasets import TrialCIFAR10
from torchvision import transforms 
from torch.utils.data import DataLoader, Subset
from training.utils.utils import log_msg

class TinyCIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset",
        num_samples: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = 0 #val_split TODO change back to val_split
        self.seed = seed

        log_msg(f"Training with {self.num_samples} samples")

        self.augmentations = TinyAugmentations(input_height=32)


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = TrialCIFAR10(self.data_dir, train=True, download=True, num_samples=self.num_samples, labels=None)
            indices = self._get_balanced_subset_indices(full_dataset)
            subset = Subset(full_dataset, indices)
            val_size = self.val_split

            self.train_dataset, self.val_dataset = self._split_dataset(subset, val_size)

            self.train_dataset = SimCLRTrainDataset(self.train_dataset, transform=self.augmentations)

        if stage == "test" or stage is None:
            self.test_dataset = TrialCIFAR10(self.data_dir, train=False, download=True, num_samples=self.num_samples, labels=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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
    
class TinyAugmentations():
    def __init__(self, input_height, jitter_strength=0.5, gaussian_blur=False):
        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        augmentation = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomApply([self.color_jitter], p=0.8), TODO this is so high that all of the values are the same afterwards 
            transforms.RandomGrayscale(p=0.2),
        ]
        self.train_transform = transforms.Compose(augmentation) # TODO add normalization #, self.final_transform])

        normalization = [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),]
        self.normalize = transforms.Compose(normalization)

    def __call__(self, sample):
        augmented_sample = self.train_transform(sample.float())
        augmented_sample = augmented_sample / 255.0
        return self.normalize(augmented_sample)


    
class SimCLRTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image1 = self.transform(image.float())
        image2 = self.transform(image.float())
        return image1, image2, label