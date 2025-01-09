import torch
import matplotlib.pyplot as plt
from collections import Counter

num_augmentations = 10 // 2
num_samples_per_class = 128
data_path = "./mnist_subset/mnist_train_subset_128_per_class_aug_10.pt"
images, labels = torch.load(data_path)

class_counts = Counter(labels.tolist())

print("Number of pairs per class:")
for class_label, count in sorted(class_counts.items()):
    print(f"Class {class_label}: {count} pairs")

def visualize_augmented_examples(images, labels, k):
    aug1, aug2 = torch.unbind(images, dim=1)
    fig, axes = plt.subplots(4, 5, figsize=(15, 6))
    offset = k * num_samples_per_class * num_augmentations
    for i in range(2):
        for j in range(5):
            ax = axes[i*2, j]
            ax.imshow(aug1[j+i*2 + offset].squeeze(), cmap="gray")
            ax.set_title(f"Class {labels[j + offset]} - aug1")
            ax.axis("off")
        for j in range(5):
            ax = axes[i*2+1, j]
            ax.imshow(aug2[j+i*2+1 + offset].squeeze(), cmap="gray")
            ax.set_title(f"Class {labels[j + offset]} - aug2")
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"mnist_visualization_augmented_{k}.png")

for k in range(10):
    visualize_augmented_examples(images, labels, k)