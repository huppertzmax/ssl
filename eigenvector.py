import time
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

k = 5  
path = "data/matrices/sparse_matrix_100_000/block/matrix.npz"
storage_path = "data/matrices/sparse_matrix_100_000/block/"

start_time = time.time()
sparse_matrix = load_npz(path)
start_time = time.time()
print(f"Matrix loaded from: {path} in: {start_time - start_time:.6f} seconds")


start_time = time.time()
eigenvalues, eigenvectors = eigsh(sparse_matrix, k=k, which='LM') 
end_time = time.time()
print(f"Runtime for {k} first eigenvectors calculation: {end_time - start_time:.6f} seconds")

print(eigenvalues)
np.save(storage_path + "eigenvalues.npy", eigenvalues)
np.save(storage_path + "eigenvectors.npy", eigenvectors)