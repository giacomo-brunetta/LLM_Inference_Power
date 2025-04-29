models_70B=(
    "meta-llama/Meta-Llama-3-70B"
    "meta-llama/Llama-2-70b-hf"
    "Qwen/Qwen2-72B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "mistralai/Mixtral-8x7B-v0.1"
)

for model in "${models_70B[@]}"; do
    python3 test_vLLM.py --model_name $model --num_gpus 4 --power
done
