models=(
    "Qwen/Qwen2-72B-Instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

set -euo pipefail

platform="${1:-cuda}"

for model in "${models[@]}"; do
    for tp_size in 4; do
       for batch_size in 16 32 64 128 -1; do
           python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --platform $platform
       done 
   done
done
