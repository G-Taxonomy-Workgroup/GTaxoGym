#!/usr/bin/env bash

homedir=$(dirname $(realpath $0))
echo homdir=${homedir}
cd $homedir

sh node_all.sh
sh node_all.sh NoFeatures
sh node_all.sh NodeDegree
sh node_all.sh NoEdges
sh node_all.sh Fragmented-k1
sh node_all.sh Fragmented-k2
sh node_all.sh Fragmented-k3
sh node_all.sh WaveletBankFiltering-lo
sh node_all.sh WaveletBankFiltering-mid
sh node_all.sh WaveletBankFiltering-hi
