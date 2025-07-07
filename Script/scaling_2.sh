set -euo pipefail

platform="${1:-cuda}"

export NUM_GPUs=4

models=(
    #"meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2-72B-Instruct"
    #"nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    #"mistralai/Mixtral-8x7B-Instruct-v0.1"
)


for model in "${models[@]}"; do
   for tp_size in 4; do
       for bs in 128 256; do
           dp_size=$(( NUM_GPUs / tp_size ))
           batch_size=$((bs * tp_size))
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
       done
    done
done
