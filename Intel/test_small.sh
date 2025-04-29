small_models=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Meta-Llama-3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "meta-llama/Llama-2-7b-hf"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    )

source /opt/intel/oneapi/setvars.sh

for model in "${small_models[@]}"; do
    echo "Running $model"
    echo "One GPU"
    python3  test_vllm.py --model_name $model --num_gpus 1 --power
    echo "Two GPUs"
    python3  test_vllm.py --model_name $model --num_gpus 2 --power
    echo "Four GPUs"
    python3  test_vllm.py --model_name $model --num_gpus 4 --power
done