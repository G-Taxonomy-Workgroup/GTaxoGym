#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

python main.py --cfg tests/configs/node/gdro.yaml --repeat 1

# Test WandB
#python main.py --cfg tests/configs/node/gdro.yaml --repeat 1 wandb.use True \
#    name_tag test_gdro optim.max_epoch 10000 metric_best auc
