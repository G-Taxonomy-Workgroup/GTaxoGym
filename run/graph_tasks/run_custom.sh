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
      script1="${commonstr} ${common_params} run_multiple_splits [0,1,2] gnn.layer_type fc_ginconv"
      script2="${commonstr} ${common_params} run_multiple_splits [3,4,5] gnn.layer_type fc_ginconv"
      script3="${commonstr} ${common_params} run_multiple_splits [6,7] gnn.layer_type fc_ginconv"
      script4="${commonstr} ${common_params} run_multiple_splits [8,9] gnn.layer_type fc_ginconv"

    elif [[ "$dataset_format" == PyG-PPI ]]; then
      script1="${commonstr} --repeat 3 seed 0 ${common_params} dataset.split_mode standard gnn.head inductive_node"
      script1="${commonstr} --repeat 3 seed 3 ${common_params} dataset.split_mode standard gnn.head inductive_node"
      script1="${commonstr} --repeat 2 seed 6 ${common_params} dataset.split_mode standard gnn.head inductive_node"
      script1="${commonstr} --repeat 2 seed 8 ${common_params} dataset.split_mode standard gnn.head inductive_node"

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
    elif [[ "$dataset_name" == ogbg-molhiv ]] || [[ "$dataset_name" == ogbg-moltox21 ]] || [[ "$dataset_name" == ogbg-molpcba ]]; then
      ogb_params="dataset.split_mode standard"
      script1="${commonstr} --repeat 3 seed 0 ${common_params} ${ogb_params}"
      script2="${commonstr} --repeat 3 seed 3 ${common_params} ${ogb_params}"
      script3="${commonstr} --repeat 2 seed 6 ${common_params} ${ogb_params}"
      script4="${commonstr} --repeat 2 seed 8 ${common_params} ${ogb_params}"
    fi

    echo $script1

    sbatch run_exp.sh "$script1"
}

for MODEL in default_gcn default_gin; do
  for PERTURB in none NoEdges FullyConnected NoFeatures NodeDegree Fragmented-k1 Fragmented-k2 Fragmented-k3 FiedlerFragmentation BandpassFiltering-hi BandpassFiltering-mid BandpassFiltering-lo RandomNodeFeatures RandomEdgeRewire; do
    test_dataset ${MODEL} PyG-TUDataset classification REDDIT-MULTI-5K ${PERTURB}
    test_dataset ${MODEL} PyG-MalNetTiny classification LocalDegreeProfile ${PERTURB}
  done
done
