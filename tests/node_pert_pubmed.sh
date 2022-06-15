#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

# test accuracy ~0.7
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1

# test accuracy ~0.62
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 perturbation.type Fragmented-k1 logging_level DETAIL

# test accuracy ~0.67
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 perturbation.type Fragmented-k3 logging_level DETAIL

# (kmeans) took 209.77sec to apply perturbation, number of edges removed = 16, test accuracy ~0.69
# (binary encoding) took 197.35sec to apply perturbation, number of edges removed = 2776, test accuracy ~0.71
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 perturbation.type ClusterSparsification logging_level DETAIL

# (band=hi) took 213.93secs, test accuracy ~0.463
# Eigvals (normalized graph laplacian) selected (n=6573): avg = 1.3155, std = 0.3177, min = 1.0000, max = 1.9855
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 perturbation.type BandpassFiltering perturbation.BandpassFiltering_band hi logging_level DETAIL

# WaveletBankFiltering(bands=[False, True, False], norm='sym')
# Took ~4 secs to perturb, test acc ~0.52, auc ~0.65
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 \
    perturbation.type WaveletBankFiltering-mid \
    logging_level DETAIL seed 1

# WaveletBankFiltering(bands=[False, False, True], norm='sym')
# Took ~4 secs to perturb, test acc ~0.75, auc ~0.87
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 \
    perturbation.type WaveletBankFiltering-lo \
    logging_level DETAIL seed 1

# WaveletBankFiltering(bands=[True, False, False], norm='sym')
# Took ~4 secs to perturb, test acc ~0.55, auc ~0.67
python main.py --cfg tests/configs/node/pubmed.yaml --repeat 1 \
    perturbation.type WaveletBankFiltering-hi \
    logging_level DETAIL seed 1
