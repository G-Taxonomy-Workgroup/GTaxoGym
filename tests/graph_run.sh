#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# OGB
python main.py --cfg tests/configs/graph/ogbg-molhiv.yaml --repeat 2

# GNNBenchmarkDataset
python main.py --cfg tests/configs/graph/cifar10.yaml
python main.py --cfg tests/configs/graph/mnist.yaml
python main.py --cfg tests/configs/graph/cluster.yaml
python main.py --cfg tests/configs/graph/pattern.yaml

# TUDataset
python main.py --cfg tests/configs/graph/collab.yaml run_multiple_splits [3]
python main.py --cfg tests/configs/graph/dd.yaml run_multiple_splits [3]
python main.py --cfg tests/configs/graph/enzymes.yaml  # run all 10 folds
python main.py --cfg tests/configs/graph/imdb-binary.yaml run_multiple_splits [0,1]
python main.py --cfg tests/configs/graph/imdb-multi.yaml run_multiple_splits [0,1]
python main.py --cfg tests/configs/graph/proteins.yaml run_multiple_splits [0,1]

### Testing runs with multiple random seeds vs. runs with multiple data splits
# Runs the second CV split (indexed from 0) 3 times with different random seeds
python main.py --cfg tests/configs/graph/nci1.yaml --repeat 3 dataset.split_index 1 run_multiple_splits [] seed 42
# Runs 3 CV splits, each with the same random seed
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7,8,9] seed 42

# Testing perturbations
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 10
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 11 perturbation.type NoFeatures
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 12 perturbation.type NodeDegree
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 13 perturbation.type NoEdges
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 14 perturbation.type FullyConnected
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 15 perturbation.type Fragmented-k1
python main.py --cfg tests/configs/graph/enzymes.yaml run_multiple_splits [] seed 16 perturbation.type Fragmented-k3
