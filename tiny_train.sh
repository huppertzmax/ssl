#!/bin/bash

date=$(date '+%Y-%m-%d %H:%M:%S')
echo $date

echo "Training custom architecture on mnist with the adam optimizer and the spectral contrastive loss and 4096 samples per class for 25 epochs"
python tiny_train.py --train_subset_name="mnist_train_subset_4096_per_class.pt" --max_epochs=25

echo "**************************************************************************************"

echo "Training custom architecture on mnist with the adam optimizer and the spectral contrastive loss and 2048 samples per class for 50 epochs"
python tiny_train.py --train_subset_name="mnist_train_subset_2048_per_class.pt" --max_epochs=50

echo "**************************************************************************************"

echo "Training custom architecture on mnist with the adam optimizer and the spectral contrastive loss and 1024 samples per class for 100 epochs"
python tiny_train.py --train_subset_name="mnist_train_subset_1024_per_class.pt" --max_epochs=100