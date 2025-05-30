models_70B=(
    "Qwen/Qwen2-72B-Instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${models_70B[@]}"; do
    python3 synthetic_test.py --model_name $model --num_gpus 4 --power
done
 