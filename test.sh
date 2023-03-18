#!/bin/sh

srun -p create --gres=gpu:3080:1 -c 4 --mem 80G python main.py --gpus 0 --config configs/xception.yaml --test