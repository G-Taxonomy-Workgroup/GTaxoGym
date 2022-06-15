#!/usr/bin/env bash

trap "kill 0" SIGINT

homedir=$(dirname $(dirname $(realpath $0)))
cd $homedir

if [[ $1 == verbose ]]; then
    echo "$(tput setaf 2)Verbose set, dumping log to stdout$(tput sgr 0)"
    redirect=''
else
    echo "$(tput setaf 2)Verbose unset, output muted." \
         "Specify 'verbose' as the first argument to show log.$(tput sgr 0)"
    redirect='> /dev/null'
fi

if [[ $CONDA_DEFAULT_ENV != gtaxogym ]]; then
    source ~/.bashrc
    conda activate gtaxogym
fi

echo homdir=${homedir}

commonstr="python main.py --cfg tests/configs/node/template_gcn.yaml --repeat 1 dataset.format"

function test_dataset {
    trap "kill 0" ERR

    dataset_format=$1
    task_type=$2

    if [[ ! -z $3 ]]; then
        namestr=" dataset.name ${3}"
    fi

    echo "Start testing ${dataset_format}, task_type=${task_type}"

    eval "${commonstr} ${dataset_format}${namestr}" \
         "dataset.split_mode standard dataset.task_type ${task_type}" \
         "out_dir tests/results/${dataset_format} ${redirect}"
}

test_dataset PyG-Actor classification
test_dataset PyG-Flickr classification
test_dataset PyG-WikiCS classification

test_dataset PyG-Planetoid classification CiteSeer
test_dataset PyG-Planetoid classification Cora
test_dataset PyG-Planetoid classification PubMed

test_dataset PyG-WikipediaNetwork classification chameleon
test_dataset PyG-WikipediaNetwork classification squirrel

test_dataset PyG-WebKB classification Cornell
test_dataset PyG-WebKB classification Texas
test_dataset PyG-WebKB classification Wisconsin

# Disabled tests due to memory limitations
#test_dataset PyG-Reddit2 classification
#test_dataset PyG-Yelp classification_multilabel

echo ALL PASSED!
