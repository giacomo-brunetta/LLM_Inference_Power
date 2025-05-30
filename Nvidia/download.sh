models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "Qwen/QwQ-32B"
    "Qwen/Qwen3-30B-A3B"
    "Qwen/Qwen2-72B-Instruct"
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "google/gemma-2-27b-it"
    "google/gemma-2-9b-it"
)
for model in "${models[@]}"; do
    echo "Downloading model: $model"
    huggingface-cli download "$model"
done