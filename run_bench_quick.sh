all_models=(
"meta-llama/Llama-2-7b-hf"
"meta-llama/Meta-Llama-3-8B"
"mistralai/Mistral-7B-v0.1"
"Qwen/Qwen2-7B"
"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
"meta-llama/Llama-2-70b-hf"
"meta-llama/Meta-Llama-3-70B"
"Qwen/Qwen2-72B"
"mistralai/Mixtral-8x7B-v0.1"
"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

small_models=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "mistralai/Mistral-7B-v0.1"
    "Qwen/Qwen2-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

big_models=(
    "mistralai/Mixtral-8x7B-v0.1"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Meta-Llama-3-70B"
    "Qwen/Qwen2-72B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)


for model in "${small_models[@]}"; do
    for num_gpus in 1 2 4; do
        python3 vLLM2.py --num_gpus $num_gpus --model_name $model --dtype "float16" --power
    done
done