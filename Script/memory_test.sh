set -euo pipefail

platform="${1:-cuda}"

model="Qwen/Qwen3-14B"

for mem_util in 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.8 0.9; do
   python3 test_text_dataset.py --model_name $model --batch_size -1 --platform $platform -dtype fp8 --memory_util $mem_util
done
