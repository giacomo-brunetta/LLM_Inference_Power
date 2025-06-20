models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"  
    "Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
)

set -euo pipefail

platform="${1:-cuda}"

for model in "${models[@]}"; do
    echo "Running $model"
    echo "One GPU"
    python  synthetic_test.py --model_name $model --num_gpus 1 --power --platform $platform
    echo "Two GPUs"
    python  synthetic_test.py --model_name $model --num_gpus 2 --power --platform $platform
    echo "Four GPUs"
    python  synthetic_test.py --model_name $model --num_gpus 4 --power --platform $platform
done 