#!/bin/bash

date=$(date '+%Y-%m-%d %H:%M:%S')
echo $date

echo "Training resnet18 on cifar10 with the lars optimizer the spectral contrastive loss"
python train.py --dataset=cifar10 --optimizer=lars --loss_type=spectral_contrastive --arch=resnet18

echo "**************************************************************************************"

echo "Training resnet18 on cifar10 with the adam optimizer the spectral contrastive loss"
python train.py --dataset=cifar10 --optimizer=adam --loss_type=spectral_contrastive --arch=resnet18

echo "**************************************************************************************"

echo "Training resnet18 on cifar10 with the sgd optimizer the spectral contrastive loss"
python train.py --dataset=cifar10 --optimizer=sgd --loss_type=spectral_contrastive --arch=resnet18