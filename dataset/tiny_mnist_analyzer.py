import torch
import matplotlib.pyplot as plt
from collections import Counter

# Load the saved subset
data_path = "./mnist_subset/mnist_val_subset_256_per_class.pt"
images, labels = torch.load(data_path)

# Count the number of elements in each class
class_counts = Counter(labels.tolist())

# Print the number of elements per class
print("Number of elements per class:")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} elements")

# Visualize one example of each class
def visualize_examples(images, labels):
    unique_classes = sorted(class_counts.keys())
    fig, axes = plt.subplots(1, len(unique_classes), figsize=(15, 3))
    for class_label, ax in zip(unique_classes, axes):
        # Find the first image of the current class
        idx = (labels == class_label).nonzero(as_tuple=True)[0][0].item()
        ax.imshow(images[idx].squeeze(), cmap="gray")
        ax.set_title(f"Class {class_label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_visualization_val.png")

visualize_examples(images, labels)
