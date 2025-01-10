#!/bin/bash

date=$(date '+%Y-%m-%d %H:%M:%S')
echo $date

echo "Linear evaluation protocol on drive-shape-38"
python tiny_evaluation.py --ckpt_path='results/driven-shape-38/epoch=16-step=2720.ckpt'

echo "Linear evaluation protocol on helpful-moon-39"
python tiny_evaluation.py --ckpt_path='results/helpful-moon-39/epoch=47-step=3840.ckpt'

echo "Linear evaluation protocol on snowy-surf-40"
python tiny_evaluation.py --ckpt_path='./results/snowy-surf-40/epoch=90-step=3640.ckpt'