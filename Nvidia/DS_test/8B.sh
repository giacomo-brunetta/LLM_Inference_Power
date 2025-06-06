models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"
    "Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
)

for model in "${models[@]}"; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name $model --tp 1 --batch_size $batch_size
    done
done
