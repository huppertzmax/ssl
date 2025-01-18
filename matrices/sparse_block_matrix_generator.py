import time
import os
import numpy as np
from argparse import ArgumentParser
from scipy.sparse import save_npz, lil_matrix, diags

def create_pair_block_sparse_matrix(n_blocks, value=1.):
    block_size = 2

    size = block_size * n_blocks
    sparse_matrix = lil_matrix((size, size))

    block = np.array([[0., value], [value, 0.]])

    for i in range(n_blocks):
        row_start = i * block_size
        col_start = i * block_size
        sparse_matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block

    return sparse_matrix.tocsr()

def create_aug_group_sparse_matrix(n_blocks, num_augs, value=1.):
    size =  num_augs * n_blocks
    sparse_matrix = lil_matrix((size, size))

    block = np.full((num_augs, num_augs), value)
    
    for i in range(n_blocks):
        row_start = i * num_augs
        col_start = i * num_augs
        sparse_matrix[row_start:row_start+num_augs, col_start:col_start+num_augs] = block

    return sparse_matrix.tocsr()

def calculate_matrix(block_type, num_samples, num_augs, n_formatted, n_blocks, value):
    start_time = time.time()
    sparse_matrix = create_pair_block_sparse_matrix(n_blocks, value) if block_type == "pair" else create_aug_group_sparse_matrix(n_blocks, num_augs, value)
    if num_samples * num_augs * 10 < 100: 
        with open(f"matrix_{block_type}_output.txt", "w") as f:
            f.write(np.array2string(sparse_matrix, 
                formatter={'float_kind': lambda x: f"{x:10.4f}"},  
                threshold=np.inf, max_line_width=np.inf))
        print(f"\nPrinted visualization of matrix under matrix_{block_type}_output.txt\n")
    end_time = time.time()
    print(f"Runtime for {block_type}-block matrix generation: {end_time - start_time:.6f} seconds")
    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Matrix sum values: {sparse_matrix.sum()}\n")
    save_npz(f"./generated/sparse_matrix_{n_formatted}/{block_type}_block/matrix_n_{num_samples}.npz", sparse_matrix)
    return sparse_matrix

def calculate_diagonal_matrices(sparse_matrix, block_type, num_samples, n_formatted):
    start_time = time.time()
    row_sums = sparse_matrix.sum(axis=1).A1
    diagonal_inv_sqrt_matrix = diags(1.0 / np.sqrt(row_sums), format="csr")
    end_time = time.time()
    print(f"Runtime for diagonal matrix generation: {end_time - start_time:.6f} seconds")
    print(f"Diagonal inverted square root matrix shape: {diagonal_inv_sqrt_matrix.shape}\n")
    save_npz(f"./generated/sparse_matrix_{n_formatted}/{block_type}_block/diagonal_inv_sqrt_matrix_n_{num_samples}.npz", diagonal_inv_sqrt_matrix)
    return diagonal_inv_sqrt_matrix

def calculate_normalized_matrix(sparse_matrix, diagonal_inv_sqrt_matrix, block_type, num_samples, n_formatted):
    start_time = time.time()
    normalized_matrix = diagonal_inv_sqrt_matrix @ sparse_matrix @ diagonal_inv_sqrt_matrix
    end_time = time.time()
    print(f"Runtime for normalized matrix generation: {end_time - start_time:.6f} seconds")
    print(f"Normalized matrix shape: {normalized_matrix.shape}\n")
    save_npz(f"./generated/sparse_matrix_{n_formatted}/{block_type}_block/normalized_matrix_n_{num_samples}.npz", normalized_matrix)

def generate(block_type, num_samples, num_augs):
    if args.block_type == "pair":
        n = args.num_samples_per_class * 2 * 10 * 100
        n_blocks = n // 2
        value = 1./n
    elif args.block_type == "aug_group":
        n = args.num_samples_per_class * args.num_aug_per_class * 10
        n_blocks = num_samples * 10
        value = 1./(num_samples * 10 *num_augs*num_augs)
    elif args.block_type == "supervised":
        n = args.num_aug_per_class * 10
        n_blocks = 10
        value = 1./(10 *num_augs*num_augs)
    
    n_formatted = f"{n:,}".replace(",", "_")
    os.makedirs(f"./generated/sparse_matrix_{n_formatted}/{args.block_type}_block", exist_ok=True)
    
    print("Start matrix generation")
    sparse_matrix = calculate_matrix(block_type, num_samples, num_augs, n_formatted, n_blocks, value)
    diagonal_inv_sqrt_matrix = calculate_diagonal_matrices(sparse_matrix, block_type, num_samples, n_formatted)
    calculate_normalized_matrix(sparse_matrix, diagonal_inv_sqrt_matrix, block_type, num_samples, n_formatted)
    print(f"Matrices saved under: generated/sparse_matrix_{n_formatted}/{block_type}_block")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_samples_per_class", type=int, default=1024)
    parser.add_argument("--num_aug_per_class", type=int, default=200)
    parser.add_argument("--block_type", type=str, default="pair", help="if not pair, will create aug_group matrix")
    args = parser.parse_args()
    print(f"Args:  {args}\n")
    generate(args.block_type, args.num_samples_per_class, args.num_aug_per_class)