#!/usr/bin/env bash

result_dir="results/graph_lr"

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

    lr_params=""
    if [[ "$dataset_name" == IMDB-BINARY ]] || [[ "$dataset_name" == CLUSTER ]]; then
      lr_params="${lr_params} optim.base_lr 0.1"
    fi

    if [[ "$dataset_format" == PyG-TUDataset ]]; then
      script1="${commonstr} ${common_params}${lr_params} run_multiple_splits [0,1,2,3]"
      script2="${commonstr} ${common_params}${lr_params} run_multiple_splits [4,5,6]"
      script3="${commonstr} ${common_params}${lr_params} run_multiple_splits [7,8,9]"
    elif [[ "$dataset_format" == PyG-GNNBenchmarkDataset ]]; then
      gbd_params=""
      if [[ "$dataset_name" == MNIST ]] || [[ "$dataset_name" == CIFAR10 ]] || [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
        gbd_params="dataset.split_mode standard"
        if [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
          gbd_params="${gbd_params} gnn.head inductive_node model.loss_fun weighted_cross_entropy"
        fi
      fi
      script1="${commonstr} --repeat 4 seed 0 ${common_params} ${gbd_params}${lr_params}"
      script2="${commonstr} --repeat 3 seed 4 ${common_params} ${gbd_params}${lr_params}"
      script3="${commonstr} --repeat 3 seed 7 ${common_params} ${gbd_params}${lr_params}"
    fi

    echo $script1

    sbatch run_exp.sh "$script1"
    sbatch run_exp.sh "$script2"
    sbatch run_exp.sh "$script3"
}

for MODEL in default_gcn; do
  for PERTURB in none; do
    test_dataset ${MODEL} PyG-TUDataset classification DD ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification ENZYMES ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification PROTEINS ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification NCI1 ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification IMDB-BINARY ${PERTURB}

    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification MNIST ${PERTURB}
    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification CIFAR10 ${PERTURB}
    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification PATTERN ${PERTURB}
    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification CLUSTER ${PERTURB}
  done
done
