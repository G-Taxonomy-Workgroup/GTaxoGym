#!/usr/bin/env bash

homedir=$(dirname $(realpath $0))
echo homdir=${homedir}
cd $homedir

if [[ ! -d ../../slurm_history ]]; then
    mkdir ../../slurm_history
else
    # remove old node runtime profiling output
    rm -rf ../../slurm_history/slurm-nrtp_*
fi

commonstr="time python main.py --cfg tests/configs/node/template_gcn.yaml --repeat 1"
result_dir="tests/results"
if [[ ! -z $1 ]]; then
    pert_type=$1
    commonstr+=" perturbation.type ${pert_type}"
    result_dir+="/pert_${pert_type}"
    echo Applied perturbation: $pert_type
else
    result_dir+="/nopert"
    echo No perturbation applied
fi
commonstr+=" dataset.format"

function test_dataset {
    dataset_format=$1
    task_type=$2
    dataset_name=$3

    job_name="nrtp_${dataset_format}"
    out_dir="${result_dir}/${dataset_format}"

    if [[ ! -z $dataset_name ]]; then
        namestr=" dataset.name ${dataset_name}"
        job_name+="_${dataset_name}"
        out_dir+="_${dataset_name}"
    fi

    script="${commonstr} ${dataset_format}${namestr} dataset.task_type ${task_type} optim.max_epoch 500 out_dir ${out_dir}"

    sbatch -J $job_name -o ../../slurm_history/slurm-%x-%A.out -t 3:55:00 --mem=32GB -A cmse --gres=gpu:v100:1 node_job_template.sb $script
}

test_dataset PyG-Actor classification
test_dataset PyG-DeezerEurope classification
test_dataset PyG-FacebookPagePage classification
test_dataset PyG-Flickr classification
test_dataset PyG-GitHub classification
test_dataset PyG-LastFMAsia classification
test_dataset PyG-WikiCS classification

test_dataset PyG-Amazon classification Computers
test_dataset PyG-Amazon classification Photo

test_dataset PyG-CitationFull classification CiteSeer
test_dataset PyG-CitationFull classification Cora
test_dataset PyG-CitationFull classification Cora_ML
test_dataset PyG-CitationFull classification DBLP
test_dataset PyG-CitationFull classification PubMed

test_dataset PyG-Coauthor classification CS
test_dataset PyG-Coauthor classification Physics

test_dataset PyG-GemsecDeezer classification_multilabel HU
test_dataset PyG-GemsecDeezer classification_multilabel HR
test_dataset PyG-GemsecDeezer classification_multilabel RO

test_dataset PyG-Planetoid classification CiteSeer
test_dataset PyG-Planetoid classification Cora
test_dataset PyG-Planetoid classification PubMed

test_dataset PyG-Twitch classification DE
test_dataset PyG-Twitch classification EN
test_dataset PyG-Twitch classification ES
test_dataset PyG-Twitch classification FR
test_dataset PyG-Twitch classification PT
test_dataset PyG-Twitch classification RU

test_dataset PyG-WikipediaNetwork classification chameleon
test_dataset PyG-WikipediaNetwork classification squirrel

test_dataset PyG-WebKB classification Cornell
test_dataset PyG-WebKB classification Texas
test_dataset PyG-WebKB classification Wisconsin

# Disabled tests due to memory limitations
#test_dataset PyG-Reddit2 classification
#test_dataset PyG-Yelp classification_multilabel
