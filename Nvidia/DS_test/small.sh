models=(
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"Qwen/Qwen3-8B"
    #"google/gemma-2-9b-it" 
    #"Qwen/Qwen3-14B"
    #"mistralai/Ministral-8B-Instruct-2410"
    #"Qwen/QwQ-32B"
    #"Qwen/Qwen3-32B"
    #"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
)

for model in "${models[@]}"; do
    for tp in 1 2 4; do
        for batch_size in 16 32 64 128 -1; do
            python test_text_dataset.py --model_name $model --tp $tp --batch_size $batch_size
        done
    done
done
