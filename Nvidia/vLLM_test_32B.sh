models_32B=(
    "Qwen/QwQ-32B"
    "Qwen/Qwen2.5-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

for model in "${models_32B[@]}"; do
    echo "Downloading $model"
    python download_model.py $model
    echo "Running $model"
    echo "One GPU"
    python  test_vLLM.py --model_name $model --num_gpus 1 --power
    echo "Two GPUs"
    python  test_vLLM.py --model_name $model --num_gpus 2 --power
    echo "Four GPUs"
    python  test_vLLM.py --model_name $model --num_gpus 4 --power
done