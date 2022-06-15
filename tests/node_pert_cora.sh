#!/usr/bin/env bash

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo "homdir=${homedir}"

python main.py --cfg tests/configs/node_pert/cora/cora_noedges.yaml --repeat 1
python main.py --cfg tests/configs/node_pert/cora/cora_nofeatures.yaml --repeat 1
python main.py --cfg tests/configs/node_pert/cora/cora_nodedegree.yaml --repeat 1
python main.py --cfg tests/configs/node_pert/cora/cora_fragmented-k1.yaml --repeat 1
python main.py --cfg tests/configs/node_pert/cora/cora_fragmented-k3.yaml --repeat 1

# (kmeans) 10 clusters, 23 edges removed (all inter), test acc ~0.76
# (binary encoding) 16 clusters, 342 edges removed (all inter), test accuracy ~0.76
python main.py --cfg tests/configs/node_pert/cora/cora_clustersparsification.yaml --repeat 1

# (num_iter=200, max_size=200) final num clusters = 105, num of edges rmvd = 2144, test acc ~0.74
# (num_iter=200, max_size=10) final num clusters = 278, num edges rmvd = 3964, test acc  ~0.74
python main.py --cfg tests/configs/node_pert/cora/cora_fiedlerfragmentation.yaml --repeat 1

# (band='hi') took 1.5 secs, test acc ~0.38
# Eigvals (normalized graph laplacian) selected (n=904): avg = 1.5835, std = 0.1933, min = 1.2945, max = 2.0000
python main.py --cfg tests/configs/node_pert/cora/cora_bandpassfiltering.yaml --repeat 1

# WaveletBankFiltering(bands=[False, True, False], norm='sym')
# Took <1 sec to perturb, test acc ~0.37, auc ~0.75
python main.py --cfg tests/configs/node_pert/cora/cora_waveletbankfiltering.yaml --repeat 1 logging_level DETAIL
