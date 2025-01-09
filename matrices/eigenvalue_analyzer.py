import numpy as np

np.set_printoptions(precision=14)

path_eigenvalues = "generated/sparse_matrix_2_048_000/block/eigenvalues_k_64.npy"
eigenvalues = np.load(path_eigenvalues)
print(f"Shape of eigenvalue list: {eigenvalues.shape}")

not_fully_unique_counter = 0

print(f"Values: {eigenvalues.shape[0]}")
unique_values = np.unique(np.round(eigenvalues, 12))
print(f"Amount of unique values: {len(unique_values)}")
print(f"Unique values: {unique_values}")