#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A UIC-HPC
#PBS -o /home/gbrun/LLM_Inference_Power/logs/output
#PBS -e /home/gbrun/LLM_Inference_Power/logs/error

./Polaris_setup.sh

cd ~/LLM_Inference_Power/Nvidia
./vLLM_test_70B.sh