import time
import numpy as np
from scipy.sparse import save_npz, lil_matrix

def create_block_sparse_matrix(n_blocks):
    block_size = 2

    size = block_size * n_blocks
    sparse_matrix = lil_matrix((size, size))

    block = np.array([[0, 1], [1, 0]])

    for i in range(n_blocks):
        row_start = i * block_size
        col_start = i * block_size
        sparse_matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block

    return sparse_matrix.tocsr()

n = 100_000  
n_formatted = formatted = f"{n:,}".replace(",", "_")
n_blocks = n // 2

start_time = time.time()
sparse_matrix = create_block_sparse_matrix(n_blocks)
end_time = time.time()
print(f"Runtime for matrix generation: {end_time - start_time:.6f} seconds")

save_npz(f"data/matrices/sparse_matrix_{n_formatted}/block/matrix.npz", sparse_matrix)
print(f"Matrix saved under: data/matrices/sparse_matrix_{n_formatted}/block/matrix.npz")