import time
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

k = 64
path = "matrices/generated/sparse_matrix_2_048_000/block/normalized_matrix_n_4096.npz"
storage_path = "matrices/generated/sparse_matrix_2_048_000/block/"

start_time = time.time()
sparse_matrix = load_npz(path)
end_time = time.time()
print(f"Matrix loaded from: {path} in: {end_time - start_time:.6f} seconds\n")


start_time = time.time()
eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='LM') 
end_time = time.time()
print(f"Runtime for {k} first eigenvectors calculation: {end_time - start_time:.6f} seconds\n")

print(eigenvalues)
print(f"Eigenvalues shape: {eigenvalues.shape}")
print(f"Eigenvectors shape: {eigenvectors.shape}")
np.save(storage_path + f"eigenvalues_k_{k}.npy", eigenvalues)
np.save(storage_path + f"eigenvectors_k_{k}.npy", eigenvectors)