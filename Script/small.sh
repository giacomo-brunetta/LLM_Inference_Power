models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"  
    "Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
)

set -euo pipefail

platform="${1:-cuda}"

for model in "${models[@]}"; do
    for tp_size in 1 2; do
       for batch_size in 16 32 64 128 -1; do
           python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --platform $platform
       done 
   done
done
