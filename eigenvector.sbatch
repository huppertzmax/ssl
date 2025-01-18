#!/bin/bash
#SBATCH --job-name=eigenvector         
#SBATCH -p lrz-cpu            
#SBATCH --qos=cpu                
#SBATCH --time=1800               
#SBATCH -o output_log_eigenvector.out
#SBATCH -e error_log_eigenvector.err   

enroot create --name ssl2 code/container/ssl.sqsh
enroot start -m code/ ssl2 << 'EOF'

cd ../code/ssl 

python eigenvector.py --num_k=32 --base_path="matrices/generated/sparse_matrix_2_048_000/aug_group_block/" --matrix_name="normalized_matrix_n_1024.npz"

python eigenvector.py --num_k=32 --base_path="matrices/generated/sparse_matrix_2_048_000/pair_block/" --matrix_name="normalized_matrix_n_1024.npz"

EOF
