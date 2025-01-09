import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import numpy as np
import os

sorted_by_label = True # Whether the dataset should be sorted by label
train_samples_per_class = 128  # Number of samples per class for training
num_augmentations = 20 # Number of augmentations per image
val_samples_per_class = 64     # Number of samples per class for validation
output_dir = "./mnist_subset"  # Directory to save the datasets
os.makedirs(output_dir, exist_ok=True)

augmentation_transformations = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.75, 1.25)),  # Rotate, translate and scale
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.025),  # Add Gaussian noise
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

def filter_indices_by_class(dataset):
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices

def create_subset_indices(train_samples_per_class, val_samples_per_class, dataset):
    class_indices = filter_indices_by_class(dataset)
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        train_indices.extend(indices[:train_samples_per_class])
        val_indices.extend(indices[train_samples_per_class:train_samples_per_class + val_samples_per_class])
    return train_indices, val_indices

def create_subsets(indices):
    images = torch.stack([mnist_train[i][0] for i in indices])
    labels = torch.tensor([mnist_train[i][1] for i in indices])
    if not sorted_by_label:
        perm = torch.randperm(len(labels))
        images = images[perm]
        labels = labels[perm]
    return TensorDataset(images, labels)

def create_augmented_dataset(dataset, transformations, n_augmentations):
    augmented_images = []
    labels = []
    for img, label in dataset:
        for _ in range(n_augmentations // 2):
            aug1 = transformations(img)
            aug2 = transformations(img)
            augmented_images.append(torch.stack([aug1, aug2]))
            labels.append(label)
    augmented_images = torch.stack(augmented_images)  # Shape: [num_samples, 2, 1, 28, 28]
    labels = torch.tensor(labels)
    return augmented_images, labels

transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root=".", train=True, download=True, transform=transform)

train_indices, val_indices = create_subset_indices(train_samples_per_class, val_samples_per_class, mnist_train)

train = create_subsets(train_indices)
val = create_subsets(val_indices)

print("Training and validation subsets created")
print(f"Training subset length {len(train)}")
print(f"Validation subset length: {len(val)}\n")

train_augmented, train_labels = create_augmented_dataset(train, augmentation_transformations, num_augmentations)
val_augmented, val_labels = create_augmented_dataset(val, augmentation_transformations, 10)

torch.save((train_augmented, train_labels), os.path.join(output_dir, f"mnist_train_subset_{train_samples_per_class}_per_class_aug_{num_augmentations}.pt"))
torch.save((val_augmented, val_labels), os.path.join(output_dir, f"mnist_val_subset_{val_samples_per_class}_per_class_aug_{num_augmentations}.pt"))

print(f"Augmented training subset created with shape: {train_augmented.shape}")
print(f"Augmented validation subset created with shape: {val_augmented.shape}")
print(f"Saved to {output_dir}")
