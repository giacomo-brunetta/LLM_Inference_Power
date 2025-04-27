#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A UIC-HPC
#PBS -o /home/gbrun/LLM_Inference_Power/logs2/output
#PBS -e /home/gbrun/LLM_Inference_Power/logs2/error

module use /soft/modulefiles
module load conda
conda activate vLLM_A100

export HF_HOME="/local/scratch"
export HF_TOKEN=""

cd ~/LLM_Inference_Power/Nvidia
./offline_test.sh