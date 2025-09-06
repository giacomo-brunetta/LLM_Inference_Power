set -euo pipefail

platform="${1:-cuda}"

export NUM_GPUs=8

models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "mistralai/Ministral-8B-Instruct-2410"
    "Qwen/Qwen3-14B"
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "Qwen/Qwen3-32B"
)

for model in "${models[@]}"; do
    for tp_size in 1 2 4; do
        for bs in 16 32 64 128; do
            dp_size=$(( NUM_GPUs / tp_size ))
            batch_size=$((bs * tp_size))
            python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8
            python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
        done
    done
done

models=(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

for model in "${models[@]}"; do
    for tp_size in 1 2 4; do
        for bs in 16 32 64 128; do
              dp_size=$(( NUM_GPUs / tp_size ))
              batch_size=$((bs * tp_size))
              python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8
        done
    done
done
