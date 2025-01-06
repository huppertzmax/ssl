#!/bin/bash
#SBATCH --job-name=tiny_pretrain          
#SBATCH -p lrz-hpe-p100x4             
#SBATCH --gres=gpu:1                  
#SBATCH --time=1800               
#SBATCH -o output_log_tiny_pretrain_ssl.out
#SBATCH -e error_log_tiny_pretrain_ssl.err   

enroot start -m code/ ssl << 'EOF'

cd ../code/ssl 

./tiny_train.sh

EOF
