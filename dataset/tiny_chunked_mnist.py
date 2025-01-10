import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random

class MultiFileLazyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.file_data = [None] * len(file_paths)  
        self.index_map = []

        for file_idx, file_path in enumerate(file_paths):
            images, _ = torch.load(file_path)
            self.index_map.extend([(file_idx, i) for i in range(len(images))])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]

        if self.file_data[file_idx] is None:
            self.file_data[file_idx] = torch.load(self.file_paths[file_idx])

        images, labels = self.file_data[file_idx]
        return images[sample_idx], labels[sample_idx]

class MultiFileDataModule(pl.LightningDataModule):
    def __init__(self, file_paths, batch_size, num_workers=4):
        super().__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = MultiFileLazyDataset(self.file_paths)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

base_path = "./mnist_subset/chunks/mnist_train_subset_1024_per_class_aug_200_chunk_"
file_paths = [f"{base_path}{i}.pt" for i in range(10)]
print(file_paths)
batch_size = 256

data_module = MultiFileDataModule(file_paths, batch_size)
data_module.setup()

train_loader = data_module.train_dataloader()

counter = 0
print(len(train_loader))
for batch in train_loader:
    images, labels = batch
    if counter % 100 == 0:
        print("Batch: ", counter)
        print(images.shape, labels.shape)
        print(labels.tolist())
        print("\n")
    counter += 1 
