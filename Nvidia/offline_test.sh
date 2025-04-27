models=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #"Qwen/Qwen2-7B-Instruct"
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    #"mistralai/Mistral-7B-Instruct-v0.1"
    #"meta-llama/Llama-2-7b-chat-hf"
    )

for model in "${smodels[@]}"; do
    python download_model.py  $model
done


for model in "${models[@]}"; do
    echo "Running $model"
    python download_model.py $model
    for gpus in 1 2 4; do
        for batch_size in 8 16 32 64 128; do
            python  offline_batch.py --model_name $model --gpus gpus --max_tokens 256 --batch_size $batch_size
        done
    done
done
