models=(
	"Qwen/Qwen3-235B-A22B-FP8"
	"deepseek-ai/DeepSeek-R1-0528"	
)

set -euo pipefail

platform="${1:-cuda}"

for model in "${models[@]}"; do
    for tp_size in 8 4 2; do
       for batch_size in 32 64 128 -1; do
           python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8 --platform $platform
       	   python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8 -ep True --platform $platform
   	done 
   done
done
