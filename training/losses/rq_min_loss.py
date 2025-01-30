import torch
import numpy as np
from training.utils.heat_kernel import compute_adjacency_matrix, compute_heat_kernel 

def rq_min_loss(out_1, out_2, heat_kernel, penalty_constrained):
    embeddings = torch.stack((out_1, out_2), dim=1)
    embeddings = embeddings.view(-1, out_1.size(1))

    if embeddings.size(0) != heat_kernel.size(0):
        heat_kernel = compute_heat_kernel(compute_adjacency_matrix(out_1.size(0), 'cuda' if torch.cuda.is_available() else 'cpu'), 1., 'cuda' if torch.cuda.is_available() else 'cpu') #TODO modify t value
        print("Shape of temporarily updated head kernel: ", heat_kernel.shape)

    trace_term = torch.trace(embeddings.T @ heat_kernel @ embeddings)
    orthogonality_term = torch.norm(embeddings.T @ embeddings - torch.eye(embeddings.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu'), p='fro')
    centering_term = torch.norm(embeddings.mean(dim=0), p=2)
    loss = -trace_term + orthogonality_term + centering_term if penalty_constrained else -trace_term
    return loss, trace_term, orthogonality_term, centering_term
