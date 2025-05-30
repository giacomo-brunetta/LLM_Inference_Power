models=(
    "Qwen/QwQ-32B"
    "Qwen/Qwen3-32B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
)

for model in "${models[@]}"; do
    echo "Running $model"
    echo "One GPU"
    python  synthetic_test.py --model_name $model --num_gpus 1 --power
    echo "Two GPUs"
    python  synthetic_test.py --model_name $model --num_gpus 2 --power
    echo "Four GPUs"
    python  synthetic_test.py --model_name $model --num_gpus 4 --power
done 