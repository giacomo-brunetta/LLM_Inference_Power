set -euo pipefail

platform="${1:-cuda}"

for tp_size in 1 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "meta-llama/Llama-3.1-8B-Instruct" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 1 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "Qwen/Qwen3-14B" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 1 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "Qwen/Qwen3-32B" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "meta-llama/Llama-3.3-70B-Instruct" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "meta-llama/Llama-3.3-70B-Instruct" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "nvidia/Llama-3_3-Nemotron-Super-49B-v1" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 2 4; do
    for bs in 16 32 64 128 256 512; do
        dp_size=$(( NUM_GPUs / tp_size ))
        batch_size=$((bs * tp_size))
        python3 data_parallel_test.py --model_name  "meta-llama/Llama-3.3-70B-Instruct" -tp $tp_size -dp $dp_size  --batch_size $batch_size --platform $platform
    done
done

for tp_size in 2 4; do
    dp_size=$(( NUM_GPUs / tp_size ))
    python3 data_parallel_test.py --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" -tp $tp_size -dp $dp_size -ep True  --batch_size -1 --platform $platform
    python3 data_parallel_test.py --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" -tp $tp_size -dp $dp_size  --batch_size -1 --platform $platform
done