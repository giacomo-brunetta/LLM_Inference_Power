small_models=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "meta-llama/Meta-Llama-3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "meta-llama/Llama-2-7b-hf"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    )


for model in "${small_models[@]}"; do
    echo "Running $model"
    echo "One GPU"
    python  test_accelerate.py --model_name $model --num_gpus 1 --power
    echo "Two GPUs"
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --gpu_ids=0,1  test_accelerate.py --model_name $model --num_gpus 2 --power
    echo "Four GPUs"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --gpu_ids=0,1,2,3  test_accelerate.py --model_name $model --num_gpus 4 --power
done