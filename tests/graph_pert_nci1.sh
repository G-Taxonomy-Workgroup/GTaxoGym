#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# acc ~0.78, auroc ~0.855
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43

# acc ~0.69, auroc ~0.75
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43 \
    perturbation.type ClusterSparsification

# acc ~0.77, auroc ~0.83
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43 \
    perturbation.type FiedlerFragmentation logging_level DEBUG

# acc ~0.75, auroc ~0.84
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43 \
    perturbation.type BandpassFiltering perturbation.BandpassFiltering_band hi

# acc ~0.81, auroc ~0.88
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43 \
    perturbation.type BandpassFiltering-lo

# acc ~0.80, auroc ~0.88
python main.py --cfg tests/configs/graph/nci1.yaml run_multiple_splits [7] seed 43 \
    perturbation.type WaveletBankFiltering \
    perturbation.WaveletBankFiltering_bands [False,False,True]
