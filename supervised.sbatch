#!/bin/bash
#SBATCH --job-name=sup         
#SBATCH -p lrz-hpe-p100x4             
#SBATCH --gres=gpu:1                  
#SBATCH --time=1800               
#SBATCH -o output_log_supervised.out
#SBATCH -e error_log_supervised.err   

enroot start -m code/ ssl << 'EOF'

cd ../code/ssl 

python supervised.py --num_epochs=10

python supervised.py --train_majority --num_epochs=10

EOF
