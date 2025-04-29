models=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "Qwen/Qwen2.5-14B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "meta-llama/Llama-2-13b-hf"
    )

for model in "${models[@]}"; do
    echo "Running $model"
    python download_model.py $model
    for gpus in 1 2 4; do
        for batch_size in 64; do
            python  offline_batch.py --model_name $model --gpus $gpus --max_tokens 256 --batch_size $batch_size
        done
    done
done
