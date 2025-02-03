import torch 
import numpy as np 
from training.utils.loss_utils import compute_adjacency_matrix, compute_graph_laplacian, compute_gaussian_kernel_matrix

def calculate_regularization_term(z, pi, sigma=0.1):
    kernel_matrix = compute_gaussian_kernel_matrix(z, sigma=sigma)
    regularization_term = torch.sum(kernel_matrix * pi)
    return regularization_term / z.shape[0]

def kernel_infonce_loss(out_1, out_2, graph_laplacian):
    z = torch.stack((out_1, out_2), dim=1)
    z = z.view(-1, out_1.size(1))

    if z.size(0) != graph_laplacian.size(0):
        graph_laplacian = compute_graph_laplacian(
            compute_adjacency_matrix(out_1.size(0), 'cuda' if torch.cuda.is_available() else 'cpu'), 
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("Shape of temporarily updated head kernel: ", graph_laplacian.shape)

    trace_term = torch.trace(z.T @ graph_laplacian @ z)
    regularization_term = calculate_regularization_term(z, graph_laplacian, 0.1) #TODO change hyperparameter
    loss = trace_term + regularization_term
    return loss, trace_term, regularization_term
