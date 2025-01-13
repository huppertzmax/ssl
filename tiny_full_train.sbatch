#!/bin/bash
#SBATCH --job-name=tiny_full         
#SBATCH -p lrz-hpe-p100x4             
#SBATCH --gres=gpu:1                  
#SBATCH --time=1800               
#SBATCH -o output_log_tiny_full_train.out
#SBATCH -e error_log_tiny_full_train.err   

enroot start -m code/ ssl << 'EOF'

cd ../code/ssl 

python tiny_full_train.py

EOF
