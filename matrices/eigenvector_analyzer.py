import numpy as np

np.set_printoptions(precision=14)

path_eigenvalues = "generated/sparse_matrix_2_048_000/block/eigenvalues_k_64.npy"
path_eigenvectors = "generated/sparse_matrix_2_048_000/block/eigenvectors_k_64.npy"
eigenvalues = np.load(path_eigenvalues)
eigenvectors = np.load(path_eigenvectors)
print(f"Shape of eigenvector matrix: {eigenvectors.shape}")

not_fully_unique_rows_counter = 0
not_fully_unique_columns_counter = 0

print(f"Rows: {eigenvectors.shape[0]}")
for i in range(eigenvectors.shape[0]):
    row = eigenvectors[i, :]
    unique_values = np.unique(np.round(row, 12))
    if (len(unique_values) < eigenvalues.shape[-1]):
        print(f"Amount of unique values in row {i + 1}: {len(unique_values)}")
        not_fully_unique_rows_counter += 1
print(f"Amount of not fully unique rows: {not_fully_unique_rows_counter}")

print(f"\nColumns: {eigenvectors.shape[1]}")
for i in range(eigenvectors.shape[1]):
    column = eigenvectors[i, :]
    unique_values = np.unique(np.round(column, 12))
    if (len(unique_values) < eigenvalues.shape[-1]):
        print(f"Amount of unique values in column {i + 1}: {len(unique_values)}")
        not_fully_unique_columns_counter += 1
print(f"Amount of not fully unique columns: {not_fully_unique_columns_counter}")