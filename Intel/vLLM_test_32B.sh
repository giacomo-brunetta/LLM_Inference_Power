models_32B=(
    "Qwen/QwQ-32B"
    "Qwen/Qwen2.5-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

for model in "${models_32B[@]}"; do
    echo "Four GPUs"
    python3  test_vLLM.py --model_name $model --num_gpus 4 --power
done