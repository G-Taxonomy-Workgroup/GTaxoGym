#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
module load cuda/11.1/cudnn/8.0 miniconda/3
conda activate gtaxogym

cd $SLURM_SUBMIT_DIR
cd ../..

echo $1
$1