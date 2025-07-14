-euo pipefail

platform="${1:-cuda}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUs=8
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


models=(
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2-72B-Instruct"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

for model in "${models[@]}"; do
   for tp_size in 1 2 4; do
       for bs in 16 32 64 128 256; do
           dp_size=$(( NUM_GPUs / tp_size ))
           batch_size=$((bs * tp_size))
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
       done
    done
done

