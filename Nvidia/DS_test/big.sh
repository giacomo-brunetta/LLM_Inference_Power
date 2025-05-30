models=(
    "Qwen/Qwen2-72B-Instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${models[@]}"; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name $model --tp 4 --batch_size $batch_size
    done
done