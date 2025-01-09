import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

#torch.manual_seed(0)

data_path = "./mnist_subset/mnist_train_subset_128_per_class_aug_10.pt"
images, labels = torch.load(data_path)
print(images.shape)
print(labels.shape)

train_dataset = TensorDataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

images, labels = next(iter(train_loader))
print("images shape: ", images.shape)
print("labels shape: ", labels.shape)
print("labels: ", labels)

aug1, aug2 = torch.unbind(images, dim=1)
print("aug1 shape: ", aug1.shape)
print("aug2 shape: ", aug2.shape)

class_counts = Counter(labels.tolist())
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} elements")

