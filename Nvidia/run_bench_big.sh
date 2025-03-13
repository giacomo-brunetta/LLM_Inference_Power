big_models=(
    "meta-llama/Meta-Llama-3-70B"
    #"mistralai/Mixtral-8x7B-v0.1"
    #"meta-llama/Llama-2-70b-hf"
    #"Qwen/Qwen2-72B"
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)


for model in "${big_models[@]}"; do
    for num_gpus in 4; do
        python3 test_vLLM.py --model_name $model --num_gpus $num_gpus --power
    done
done