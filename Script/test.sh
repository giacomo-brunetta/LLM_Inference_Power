models=(
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"Qwen/Qwen3-8B"
    #"Qwen/Qwen3-14B"
    "mistralai/Ministral-8B-Instruct-2410"
    "Qwen/Qwen3-32B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "Qwen/Qwen2-72B-Instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

set -euo pipefail

platform="${1:-cuda}"

for model in "${models[@]}"; do
    for batch_size in 16 32 64 128 -1; do
         python3 test_text_dataset.py --model_name $model -tp 1 --batch_size $batch_size --platform $platform
	 python3 test_text_dataset.py --model_name $model -tp 1 --batch_size $batch_size --platform $platform --data_type fp8
    done
done
