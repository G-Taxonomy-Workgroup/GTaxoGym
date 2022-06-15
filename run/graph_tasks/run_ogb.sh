#!/usr/bin/env bash

result_dir="results/graph"

function test_dataset {
    cfg_name=$1
    dataset_format=$2
    task_type=$3
    dataset_name=$4
    perturbation=$5

    commonstr="python main.py --cfg configs/graph/${cfg_name}.yaml"
    out_dir="${result_dir}/${perturbation}/${dataset_format}_${dataset_name}"
    namestr="dataset.name ${dataset_name}"
    common_params="dataset.format ${dataset_format} ${namestr} dataset.task_type ${task_type} out_dir ${out_dir} perturbation.type ${perturbation}"

    if [[ "$dataset_format" == PyG-TUDataset ]] || [[ "$dataset_format" == PyG-MalNetTiny ]] || [[ "$dataset_format" == nx ]]; then
      script1="${commonstr} ${common_params} run_multiple_splits [0,1,2]"
      script2="${commonstr} ${common_params} run_multiple_splits [3,4,5]"
      script3="${commonstr} ${common_params} run_multiple_splits [6,7]"
      script4="${commonstr} ${common_params} run_multiple_splits [8,9]"
    elif [[ "$dataset_format" == PyG-GNNBenchmarkDataset ]]; then
      gbd_params=""
      if [[ "$dataset_name" == MNIST ]] || [[ "$dataset_name" == CIFAR10 ]] || [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
        gbd_params="dataset.split_mode standard"
        if [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
          gbd_params="${gbd_params} gnn.head inductive_node model.loss_fun weighted_cross_entropy"
        fi
      fi
      script1="${commonstr} --repeat 3 seed 0 ${common_params} ${gbd_params}"
      script2="${commonstr} --repeat 3 seed 3 ${common_params} ${gbd_params}"
      script3="${commonstr} --repeat 2 seed 6 ${common_params} ${gbd_params}"
      script4="${commonstr} --repeat 2 seed 8 ${common_params} ${gbd_params}"
    elif [[ "$dataset_name" == ogbg-molhiv ]] || [[ "$dataset_name" == ogbg-moltox21 ]]; then
      ogb_params="dataset.split_mode standard"
      script1="${commonstr} --repeat 3 seed 0 ${common_params} ${ogb_params}"
      script2="${commonstr} --repeat 3 seed 3 ${common_params} ${ogb_params}"
      script3="${commonstr} --repeat 2 seed 6 ${common_params} ${ogb_params}"
      script4="${commonstr} --repeat 2 seed 8 ${common_params} ${ogb_params}"
    fi

    echo $script1

    sbatch run_exp.sh "$script1"
    sbatch run_exp.sh "$script2"
    sbatch run_exp.sh "$script3"
    sbatch run_exp.sh "$script4"
}

for MODEL in default_gcn default_gin; do
  for PERTURB in none NoEdges NoFeatures NodeDegree FullyConnected Fragmented-k1 Fragmented-k2 Fragmented-k3 FiedlerFragmentation BandpassFiltering-hi BandpassFiltering-mid BandpassFiltering-lo; do
    test_dataset ${MODEL} OGB classification ogbg-molhiv ${PERTURB}
    test_dataset ${MODEL} OGB classification_multilabel ogbg-moltox21 ${PERTURB}
  done
done