models_14B=(
    "Qwen/Qwen2.5-14B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "meta-llama/Llama-2-13b-hf"
)

for model in "${models_14B[@]}"; do
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