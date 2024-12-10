#!/bin/bash
#SBATCH --job-name=train_job          
#SBATCH -p lrz-dgx-1-p100x8              
#SBATCH --gres=gpu:1                  
#SBATCH --time=660               
#SBATCH -o output_log_pretrain_ssl.out
#SBATCH -e error_log_pretrain_ssl.err   

enroot create --name ssl_container2 ./code/container/kernel-infonce.sqsh
enroot start -m code/ ssl_container2 << 'EOF'

cd ../code/Kernel-InfoNCE

./train.sh

EOF
