#!/usr/bin/env bash

function test_dataset {
    cfg_name=$1
    dataset_format=$2
    task_type=$3
    dataset_name=$4
    perturbation=$5

    commonstr="python main.py --cfg configs/graph/${cfg_name}.yaml"
    namestr="dataset.name ${dataset_name}"
    common_params="dataset.format ${dataset_format} ${namestr} dataset.task_type ${task_type} perturbation.type ${perturbation}"

    for ACT in relu prelu; do
      for STAGE in skipsum skipconcat; do
        for AGG in add mean max; do
          result_dir="results/graph_sweeps/${ACT}_${STAGE}_${AGG}"
          out_dir="${result_dir}/${perturbation}/${dataset_format}_${dataset_name}"
          sweepstr="gnn.act ${ACT} gnn.stage_type ${STAGE} gnn.agg ${AGG} out_dir ${out_dir}"
          if [[ "$dataset_format" == PyG-TUDataset ]] || [[ "$dataset_format" == nx ]]; then
            script1="${commonstr} ${common_params} ${sweepstr} run_multiple_splits [0,1,2]"
          elif [[ "$dataset_format" == PyG-GNNBenchmarkDataset ]]; then
            gbd_params=""
            if [[ "$dataset_name" == MNIST ]] || [[ "$dataset_name" == CIFAR10 ]] || [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
              gbd_params="dataset.split_mode standard"
              if [[ "$dataset_name" == PATTERN ]] || [[ "$dataset_name" == CLUSTER ]]; then
                gbd_params="${gbd_params} gnn.head inductive_node model.loss_fun weighted_cross_entropy"
              fi
            fi
            script1="${commonstr} --repeat 3 seed 0 ${common_params} ${gbd_params} ${sweepstr}"
          fi

          echo $script1

          sbatch run_exp.sh "$script1"
        done
      done
    done
}

for MODEL in default_gin; do
  for PERTURB in none NoEdges FullyConnected NoFeatures NodeDegree Fragmented-k1 Fragmented-k2 Fragmented-k3 FiedlerFragmentation BandpassFiltering-hi BandpassFiltering-mid BandpassFiltering-lo; do
    test_dataset ${MODEL} PyG-TUDataset classification ENZYMES ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification PROTEINS ${PERTURB}
    test_dataset ${MODEL} PyG-TUDataset classification IMDB-BINARY ${PERTURB}

    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification CIFAR10 ${PERTURB}
    test_dataset ${MODEL} PyG-GNNBenchmarkDataset classification CLUSTER ${PERTURB}
  done
done
