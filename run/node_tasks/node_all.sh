#!/usr/bin/env bash
# This is a shell script for submitting all node level tasks evaluation jobs
#
# ARGUMENTS:
#   Perturbation type, leave empty if no perturbation
#
# EXAMPLE:
#   $ sh node_all.sh  # submit all evaluations on the original datasets
#   $ sh node_all.sh NoEdges  # submit evaluations on with NoEdges perturbation
#
# Instruction for adding new datasets:
#   - Profile runtime and determine the size of the dataset, 'small' or 'large'
#       A 'small' dataset is the the one that could be finished within 25 sec
#       for 100 epoch, which means during the production (5,000 epoch with 10
#       repetitions), it could be finished within 3.5 hours
#   - Determine the task type: classification vs. classification_multilabel
#   - Determine split type: if predefined split is available, use 'standard',
#       oatherwise use 'random'
#   - Also specify specific dataset 'name' as the final argument if applicable
#
# Example for adding new dataset:
#   $ eval_dataset small PyG-Planetoid classification standard Cora
#   Evaluate the PyG-Planetoid dataset ('Cora' in particular), which is a
#       'small' dataset, with multiclass 'classification' tasks, and has
#       predefined 'standard' splits


homedir=$(dirname $(realpath $0))
cd $homedir

if [[ ! -d ../../slurm_history ]]; then
    mkdir ../../slurm_history
fi

commonstr="time python main.py --cfg configs/node_gcn_v2.yaml"
result_dir="results/node"
if [[ ! -z $1 ]]; then
    pert_type=$1
    pert_type_str=" perturbation.type ${pert_type}"
    result_dir+="/pert_${pert_type}"
    echo Applied perturbation: $pert_type
else
    result_dir+="/nopert"
    echo No perturbation applied
fi

function eval_dataset {
    job_size=$1
    dataset_format=$2
    task_type=$3
    split_mode=$4
    dataset_name=$5

    job_name="eval_${dataset_format}"
    out_dir="${result_dir}/${dataset_format}"

    if [[ ! -z $dataset_name ]]; then
        namestr=" dataset.name ${dataset_name}"
        job_name+="_${dataset_name}"
        out_dir+="_${dataset_name}"
    else
        namestr=""
    fi

    settings="dataset.format ${dataset_format}${namestr} ${pert_type_str} "
    settings+="dataset.split_mode ${split_mode} dataset.task_type ${task_type} out_dir ${out_dir}"

    if [[ $job_size == small ]]; then
        script="${commonstr} --repeat 10 ${settings}"
        sbatch -J $job_name --nice=100 node_job_template_single.sb $script
    else
        script="${commonstr} --repeat 1 ${settings}"
        # Prioritize large jobs by nicing to 99 (default 100)
        sbatch -J $job_name --nice=99 node_job_template_multi.sb $script
    fi
}

# standard splits
eval_dataset small PyG-Actor classification standard
eval_dataset large PyG-Flickr classification standard
eval_dataset small PyG-WikiCS classification standard

eval_dataset small PyG-Planetoid classification standard CiteSeer
eval_dataset small PyG-Planetoid classification standard Cora
eval_dataset small PyG-Planetoid classification standard PubMed

eval_dataset small PyG-WikipediaNetwork classification standard chameleon
eval_dataset small PyG-WikipediaNetwork classification standard squirrel

eval_dataset small PyG-WebKB classification standard Cornell
eval_dataset small PyG-WebKB classification standard Texas
eval_dataset small PyG-WebKB classification standard Wisconsin

# random splits
eval_dataset small PyG-DeezerEurope classification random
eval_dataset small PyG-FacebookPagePage classification random
eval_dataset small PyG-GitHub classification random
eval_dataset small PyG-LastFMAsia classification random

eval_dataset small PyG-Amazon classification random Computers
eval_dataset small PyG-Amazon classification random Photo

eval_dataset small PyG-CitationFull classification random CiteSeer
eval_dataset large PyG-CitationFull classification random Cora
eval_dataset small PyG-CitationFull classification random Cora_ML
eval_dataset small PyG-CitationFull classification random DBLP
eval_dataset small PyG-CitationFull classification random PubMed

eval_dataset large PyG-Coauthor classification random CS
eval_dataset large PyG-Coauthor classification random Physics

eval_dataset small PyG-Twitch classification random DE
eval_dataset small PyG-Twitch classification random EN
eval_dataset small PyG-Twitch classification random ES
eval_dataset small PyG-Twitch classification random PT

# Excluded due to poor base line performance
#eval_dataset small PyG-Twitch classification random FR
#eval_dataset small PyG-Twitch classification random RU
#eval_dataset small PyG-GemsecDeezer classification_multilabel random HU
#eval_dataset large PyG-GemsecDeezer classification_multilabel random HR
#eval_dataset small PyG-GemsecDeezer classification_multilabel random RO
