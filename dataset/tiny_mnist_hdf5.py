import torch
import h5py
from torch.utils.data import Dataset

class TinyMNISTHDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.h5file = h5py.File(h5_file_path, "r")
        self.images = self.h5file["images"]
        self.labels = self.h5file["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label

    def __del__(self):
        self.h5file.close()