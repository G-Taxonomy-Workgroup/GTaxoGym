#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# acc ~0.33, auroc ~0.66
python main.py --cfg configs/graph/default_gcn.yaml \
    dataset.format PyG-TUDataset \
    dataset.name ENZYMES \
    dataset.task_type classification \
    out_dir "tests/results/enzymes" \
    perturbation.type BandpassFiltering \
    perturbation.BandpassFiltering_band lo \
    run_multiple_splits [4,]

# acc ~0.38, auroc ~0.74
python main.py --cfg configs/graph/default_gcn.yaml \
    dataset.format PyG-TUDataset \
    dataset.name ENZYMES \
    dataset.task_type classification \
    out_dir "tests/results/enzymes" \
    perturbation.type WaveletBankFiltering \
    perturbation.WaveletBankFiltering_bands [False,True,False] \
    run_multiple_splits [4,]
