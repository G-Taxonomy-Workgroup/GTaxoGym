#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

python main.py --cfg tests/configs/node/wikics.yaml --repeat 3
