#!/bin/bash

echo "Training resnet18 on cifar10 with the lars optimizer the nt_xent loss"
python train.py --dataset=cifar10 --optimizer=lars --loss_type=nt_xent --arch=resnet18

echo "**************************************************************************************"

echo "Training resnet18 on cifar10 with the lars optimizer the origin loss"
python train.py --dataset=cifar10 --optimizer=lars --loss_type=origin --arch=resnet18