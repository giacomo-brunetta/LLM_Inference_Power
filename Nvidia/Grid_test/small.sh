models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"  
    "Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
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