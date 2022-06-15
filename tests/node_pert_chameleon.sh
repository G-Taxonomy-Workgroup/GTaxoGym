#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# test acc ~0.35, auc ~0.73
python main.py --cfg tests/configs/node/chameleon.yaml --repeat 1

# (takes ~1.04sec to perturb, removed edges = 151 / 12681) test acc ~0.48, auroc ~0.765
python main.py --cfg tests/configs/node/chameleon.yaml --repeat 1 perturbation.type ClusterSparsification

# (takes ~147sec to perturb, final num clusters = 774, removed edges = 30869 / 36101) test acc ~0.48, auc ~0.78
python main.py --cfg tests/configs/node/chameleon.yaml --repeat 1 perturbation.type FiedlerFragmentation logging_level DEBUG
