#!/bin/bash

echo "Finetuning feasible-hill-60 with resnet50 architecture on cifar10"
python linear_evaluation.py --dataset=cifar10 --ckpt_path='./Kernel-InfoNCE/feasible-hill-60/checkpoints/epoch=94-step=31635.ckpt' --arch=resnet50

echo "Finetuning vibrant-paper-61 with resnet50 architecture on cifar10"
python linear_evaluation.py --dataset=cifar10 --ckpt_path='./Kernel-InfoNCE/vibrant-paper-61/checkpoints/epoch=95-step=31968.ckpt' --arch=resnet50

echo "Finetuning dauntless-surf-62 with resnet50 architecture on cifar10"
python linear_evaluation.py --dataset=cifar10 --ckpt_path='./Kernel-InfoNCE/dauntless-surf-62/checkpoints/epoch=93-step=31302.ckpt' --arch=resnet50

echo "Finetuning eternal-planet-63 with resnet50 architecture on cifar10"
python linear_evaluation.py --dataset=cifar10 --ckpt_path='./Kernel-InfoNCE/eternal-planet-63/checkpoints/epoch=96-step=32301.ckpt' --arch=resnet50