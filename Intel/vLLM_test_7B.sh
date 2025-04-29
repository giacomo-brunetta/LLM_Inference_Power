models_7B=(
    #"meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )


for model in "${models_7B[@]}"; do
    python3  test_vLLM.py --model_name $model --num_gpus 1 --power
    #echo "Two GPUs"
    #python  test_vLLM.py --model_name $model --num_gpus 2 --power
    #echo "Four GPUs"
    #python  test_vLLM.py --model_name $model --num_gpus 4 --power
done