models=(
    "Qwen/QwQ-32B"
    "Qwen/Qwen3-32B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
)

for model in "${models[@]}"; do
    for tp in 2 4; do
        for batch_size in 16 32 64 128 -1; do
            python test_text_dataset.py --model_name $model --tp $tp --batch_size $batch_size
        done
    done
done
