set -euo pipefail

platform="${1:-cuda}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUs=8
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

models=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

for model in "${models[@]}"; do
   for tp_size in 1 2 4; do
       for bs in 16 32 64 128; do
           dp_size=$(( NUM_GPUs / tp_size ))
           batch_size=$((bs * tp_size))
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -dtype fp8 -s weak
           python3 data_parallel_test.py --model_name  $model -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform -s weak
       done
    done
done

