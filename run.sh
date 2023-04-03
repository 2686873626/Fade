# !/bin/sh

#srun -p create -w irip-cluster-compute-3 --gres=gpu:3080:1 -c 8 --mem 160G python main.py --gpus 0  --config configs/xception.yaml
# srun -p create --gres=gpu:3080:4 -c 8 --mem 80G python main.py --gpus 0 1 2 3 --config configs/xception.yaml
# srun -p create --gres=gpu:3080:8 -c 12 --mem 160G python main.py --gpus 0 1 2 3 4 5 6 7 --config configs/xception.yaml
srun -p create -w irip-cluster-compute-3 --gres=gpu:3080:4 -c 8 --mem 160G python main.py --gpus 0 1 2 3 --config configs/xception.yaml
# srun -p create --gres=gpu:3080:4 -c 8 --mem 120G python main.py --gpus 0 1 2 3 --config configs/xcep.yaml