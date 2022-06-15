#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# test auc ~0.99
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Computers

# test auc ~0.5
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Computers perturbation.type NoFeaturesNoEdges

# test auc ~0.68
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Computers perturbation.type NoFeaturesFragk1 optim.max_epoch 5000

# test auc ~0.99
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Photo

# test auc ~0.5
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Photo perturbation.type NoFeaturesNoEdges

# test auc ~0.71
python main.py --cfg tests/configs/node/amazon.yaml --repeat 1 dataset.name Photo perturbation.type NoFeaturesFragk1 optim.max_epoch 5000
