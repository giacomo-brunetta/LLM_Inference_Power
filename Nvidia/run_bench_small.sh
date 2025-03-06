small_models=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)


for model in "${small_models[@]}"; do
    for num_gpus in 1 2 4; do
        python3 vLLM_multiple.py --model_name $model --num_gpus $num_gpus --power
    done
done