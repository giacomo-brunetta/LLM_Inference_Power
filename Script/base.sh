set -euo pipefail

platform="${1:-cuda}"

export NUM_GPUs=1
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0


models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "mistralai/Ministral-8B-Instruct-2410"
    "Qwen/Qwen3-14B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "Qwen/Qwen3-32B"
)

for model in "${models[@]}"; do
    for bs in 16 32 64 128; do
        python3 data_parallel_test.py --model_name  $model  --batch_size $bs --platform $platform -dtype fp8
        python3 data_parallel_test.py --model_name  $model  --batch_size $bs --platform $platform
    done
done
    
models=(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    "Qwen/Qwen2-72B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${models[@]}"; do
    for bs in 16 32 64 128; do
          python3 data_parallel_test.py --model_name  $model --batch_size $bs --platform $platform -dtype fp8
	  python3 data_parallel_test.py --model_name  $model  --batch_size $bs --platform $platform
    done
done
