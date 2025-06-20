models=(
    "Qwen/QwQ-32B"
    "Qwen/Qwen3-32B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
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