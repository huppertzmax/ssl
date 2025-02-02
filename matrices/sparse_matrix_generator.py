import time
import os 
import numpy as np
from scipy.sparse import random, save_npz, csr_matrix

from sparse_block_matrix_generator import calculate_diagonal_matrices, calculate_normalized_matrix

n_block = 204_800 
n = n_block * 10
n_formatted = formatted = f"{n:,}".replace(",", "_")
density = 0.0001
target_sum = 1.0 / 10

sparse_matrix = csr_matrix((n, n))

for i in range(10):
    print(f"################# Start iteration {i} #################")
    start_time = time.time()
    rng = np.random.default_rng(seed=i) 
    sparse_matrix_block = random(n_block, n_block, density=density, format='csr', random_state=rng)
    end_time = time.time()
    print(f"Runtime for matrix generation: {end_time - start_time:.6f} seconds")
    print(f"Matrix sum values: {sparse_matrix_block.sum()}\n")

    num_nonzero = sparse_matrix_block.nnz
    print("Number of non-zero elements: ", num_nonzero)
    random_values = rng.random(num_nonzero)  
    random_values = (random_values / random_values.sum()) * target_sum  
    sparse_matrix_block.data = random_values
    print(f"Matrix sum values: {sparse_matrix_block.sum()}\n")

    rows, cols = sparse_matrix_block.nonzero()
    values = sparse_matrix_block.data

    rows = rows + i * n_block
    cols = cols + i * n_block

    sparse_matrix_block = csr_matrix((values, (rows, cols)), shape=(n, n))

    sparse_matrix = sparse_matrix + sparse_matrix_block

print(f"Matrix sum values: {sparse_matrix.sum()}")
print(f"Matrix non-zero values: {sparse_matrix.nnz}\n")

start_time = time.time()
os.makedirs(f"./generated/sparse_matrix_{n_formatted}/supervised_random_block", exist_ok=True)
save_npz(f"./generated/sparse_matrix_{n_formatted}/supervised_random_block/matrix_random_density_{density}.npz", sparse_matrix)
end_time = time.time()
print(f"Runtime for saving matrix: {end_time - start_time:.6f} seconds")
print(f"Matrix saved under: generated/sparse_matrix_{n_formatted}/supervised_random_block/matrix_random_density_{density}.npz")

diagonal_inv_sqrt_matrix = calculate_diagonal_matrices(sparse_matrix, "supervised_random", 204000, n_formatted, density)
calculate_normalized_matrix(sparse_matrix, diagonal_inv_sqrt_matrix, "supervised_random", 204800, n_formatted, density)