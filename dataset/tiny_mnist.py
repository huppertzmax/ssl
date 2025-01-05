import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from training.utils.utils import log_msg

class TinyMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./dataset/mnist_subset/",
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.augmentations = TinyAugmentations(input_height=28)


    def prepare_data(self):
        self.train_dataset = torch.load(self.data_dir + "mnist_train_subset_1024_per_class.pt")
        self.val_dataset = torch.load(self.data_dir + "mnist_val_subset_256_per_class.pt")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TensorDataset(*self.train_dataset)
            self.train_dataset = AugmentedDataset(self.train_dataset, transform=self.augmentations)
            self.val_dataset = TensorDataset(*self.val_dataset)
            self.val_dataset = AugmentedDataset(self.val_dataset, transform=self.augmentations)

        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transforms.ToTensor())


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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
            #transforms.RandomGrayscale(p=0.2),
        ]
        self.train_transform = transforms.Compose(augmentation) # TODO add normalization #, self.final_transform])

        #normalization = [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),]
        #self.normalize = transforms.Compose(normalization)

    def __call__(self, sample):
        #augmented_sample = self.train_transform(sample.float())
        return self.train_transform(sample.float())
        #augmented_sample = augmented_sample / 255.0
        #return self.normalize(augmented_sample)


    
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image1 = self.transform(image.float())
        image2 = self.transform(image.float())
        '''
        plt.figure(figsize=(5, 5))
        plt.imshow(image1.squeeze(), cmap="gray")
        plt.title(label.item())
        plt.axis("off")
        plt.savefig("img1_visualization.png")

        plt.figure(figsize=(5, 5))
        plt.imshow(image2.squeeze(), cmap="gray")
        plt.title(label.item())
        plt.axis("off")
        plt.savefig("img2_visualization.png")
        log_msg("Image 1 and Image 2 saved in img1_visualization.png and img2_visualization.png")
        exit()
        '''

        return image1, image2, label