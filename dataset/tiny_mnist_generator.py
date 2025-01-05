import torch
from torchvision import datasets, transforms
import numpy as np
import os

train_samples_per_class = 1024  # Number of samples per class for training
val_samples_per_class = 256     # Number of samples per class for validation
output_dir = "./mnist_subset"  # Directory to save the datasets

os.makedirs(output_dir, exist_ok=True)

transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root=".", train=True, download=True, transform=transform)

class_indices = {i: [] for i in range(10)}
for idx, (image, label) in enumerate(mnist_train):
    class_indices[label].append(idx)

train_indices = []
val_indices = []
for label, indices in class_indices.items():
    np.random.shuffle(indices)
    train_indices.extend(indices[:train_samples_per_class])
    val_indices.extend(indices[train_samples_per_class:train_samples_per_class + val_samples_per_class])

train_images = torch.stack([mnist_train[i][0] for i in train_indices])
train_labels = torch.tensor([mnist_train[i][1] for i in train_indices])

val_images = torch.stack([mnist_train[i][0] for i in val_indices])
val_labels = torch.tensor([mnist_train[i][1] for i in val_indices])

train_perm = torch.randperm(len(train_labels))
train_images = train_images[train_perm]
train_labels = train_labels[train_perm]

val_perm = torch.randperm(len(val_labels))
val_images = val_images[val_perm]
val_labels = val_labels[val_perm]

torch.save((train_images, train_labels), os.path.join(output_dir, f"mnist_train_subset_{train_samples_per_class}_per_class.pt"))
torch.save((val_images, val_labels), os.path.join(output_dir, f"mnist_val_subset_{val_samples_per_class}_per_class.pt"))

print(f"Training subset created with shape: {train_images.shape}, {train_labels.shape}")
print(f"Validation subset created with shape: {val_images.shape}, {val_labels.shape}")
print(f"Saved to {output_dir}")
