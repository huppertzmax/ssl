import torch 
import numpy as np

def compute_adjacency_matrix(batch_size, device="cuda"):
    value = 1.0 / (batch_size * 2.0) #TODO check if should be 1. or 1./(2*batchsize)
    block = torch.tensor([[0., value], [value, 0.]], dtype=torch.float32)
    blocks = [block for _ in range(batch_size)]
    adj_matrix = torch.block_diag(*blocks)
    print("Shape of adjacency matrix: ",adj_matrix.shape)
    return adj_matrix.to(device)

def compute_heat_kernel(adj_matrix, t, device="cuda"):
    degree_matrix = torch.diag(adj_matrix.sum(dim=1)).to(device)
    d_inv = torch.diag(1.0 / degree_matrix.diag()).to(device)
    norm_laplacian = torch.eye(adj_matrix.shape[0]).to(device) - d_inv @ adj_matrix

    eigvals, eigvecs = torch.linalg.eigh(norm_laplacian)
    heat_kernel = eigvecs @ torch.diag(torch.exp(-t * eigvals)) @ eigvecs.T
    print("Shape of eigenvectors: ", eigvecs.shape)
    print("Shape of eigenvalues: ", eigvecs.shape)
    print("Shape of heat_kernel: ", heat_kernel.shape)
    return heat_kernel.to(device)
