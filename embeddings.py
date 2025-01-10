import torch
import os  
import time
from torch.utils.data import DataLoader, TensorDataset

from training.models.tiny_mnist_backbone import TinyMNISTBackbone
from training.utils.ckpt_loading import update_ckpt_dict

torch.manual_seed(0)

num_augmentations = 20
num_samples_per_class = 128
ckpt_path = "/dss/dsshome1/lxc03/apdl006/thesis/code/ssl/results/magic-paper-99/epoch=20-step=3360.ckpt"
data_path = f"./dataset/mnist_subset/mnist_train_subset_{num_samples_per_class}_per_class_aug_{num_augmentations}.pt"
storage_path = "./results/embeddings/magic-paper-99/"
os.makedirs(storage_path, exist_ok=True)

start_time = time.time()
print(f"Start loading dataset from path: {data_path}")
images, labels = torch.load(data_path)
dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=512, shuffle=False)
end_time = time.time()
print(f"Loaded dataset in: {end_time - start_time:.6f} seconds\n")

start_time = time.time()
print(f"Start loading model from checkpoint: {ckpt_path}")
backbone = TinyMNISTBackbone()
backbone.load_state_dict(update_ckpt_dict(ckpt_path))
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False
end_time = time.time()
print(f"Loaded model in: {end_time - start_time:.6f} seconds\n")


start_time = time.time()
print(f"Start calculating neural network embedding in {len(loader)} batches")
embeddings = []
batch_counter = 0
for batch in loader:
    images, _ = batch 
    images = images.view(images.size(dim=0)*2, 1, 28, 28)
    embedding = backbone(images)
    print(f"Shape of embedding of batch: {batch_counter} is: {embedding.shape}")
    batch_counter += 1
    embeddings.append(embedding)

embeddings = torch.cat(embeddings)
print(f"\nShape of produced embedding: {embeddings.shape}")
end_time = time.time()
print(f"Calculated embedding in: {end_time - start_time:.6f} seconds\n")

file_path = storage_path + f"embeddings_{num_samples_per_class}_{num_augmentations}.pt"
torch.save(embeddings, file_path)
print(f"Stored embedding under {file_path}")