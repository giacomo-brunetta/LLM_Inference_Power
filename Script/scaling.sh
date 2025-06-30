set -euo pipefail

platform="${1:-cuda}"

export NUM_GPUs=4
export HIP_VISIBLE_DEVICES=0,1,2,3

models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"
    "Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
)

for model in "${models[@]}"; do   
   for tp_size in 1 2 4; do
      for bs in 16 32 64 128 256 512; do
         dp_size=$(( NUM_GPUs / tp_size ))
         batch_size=$((bs * tp_size))
	 python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
      done
   done
done

export NUM_GPUs=8i
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

models=(
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2-72B-Instruct"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

for model in "${models[@]}"; do   
   for tp_size in 1 2 4 8; do
       for bs in 16 32 64 128 256 512; do
           dp_size=$(( NUM_GPUs / tp_size ))
           batch_size=$((bs * tp_size))
	   python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8
       done
    done
done
