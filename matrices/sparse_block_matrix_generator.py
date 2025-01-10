import time
import os
import numpy as np
from scipy.sparse import save_npz, lil_matrix, diags

def create_block_sparse_matrix(n_blocks, value=1.):
    block_size = 2

    size = block_size * n_blocks
    sparse_matrix = lil_matrix((size, size))

    block = np.array([[0., value], [value, 0.]])

    for i in range(n_blocks):
        row_start = i * block_size
        col_start = i * block_size
        sparse_matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block

    return sparse_matrix.tocsr()

num_samples = 4096
n = num_samples * 10 * 2 * 25  
n_formatted = formatted = f"{n:,}".replace(",", "_")
n_blocks = n // 2
value = 1./n

os.makedirs(f"./generated/sparse_matrix_{n_formatted}/block", exist_ok=True)

start_time = time.time()
sparse_matrix = create_block_sparse_matrix(n_blocks, value)
end_time = time.time()
print(f"Runtime for matrix generation: {end_time - start_time:.6f} seconds")
print(f"Matrix shape: {sparse_matrix.shape}")
print(f"Matrix sum values: {sparse_matrix.sum()}\n")
save_npz(f"./generated/sparse_matrix_{n_formatted}/block/matrix_n_{num_samples}.npz", sparse_matrix)

start_time = time.time()
row_sums = sparse_matrix.sum(axis=1).A1
diagonal_inv_sqrt_matrix = diags(1.0 / np.sqrt(row_sums), format="csr")
end_time = time.time()
print(f"Runtime for diagonal matrix generation: {end_time - start_time:.6f} seconds")
print(f"Diagonal inverted square root matrix shape: {diagonal_inv_sqrt_matrix.shape}\n")
save_npz(f"./generated/sparse_matrix_{n_formatted}/block/diagonal_inv_sqrt_matrix_n_{num_samples}.npz", diagonal_inv_sqrt_matrix)

start_time = time.time()
normalized_matrix = diagonal_inv_sqrt_matrix @ sparse_matrix @ diagonal_inv_sqrt_matrix
end_time = time.time()
print(f"Runtime for normalized matrix generation: {end_time - start_time:.6f} seconds")
print(f"Normalized matrix shape: {normalized_matrix.shape}\n")
save_npz(f"./generated/sparse_matrix_{n_formatted}/block/normalized_matrix_n_{num_samples}.npz", normalized_matrix)

print(f"Matrices saved under: generated/sparse_matrix_{n_formatted}/block")