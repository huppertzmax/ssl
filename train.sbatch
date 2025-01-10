#!/bin/bash
#SBATCH --job-name=pretrain          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=1800               
#SBATCH -o output_log_pretrain_ssl.out
#SBATCH -e error_log_pretrain_ssl.err   

enroot create --name ssl3 ./code/container/ssl.sqsh
enroot start -m code/ ssl3 << 'EOF'

cd ../code/ssl 

./train.sh

EOF
