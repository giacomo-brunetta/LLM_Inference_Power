models=(
	#"mistralai/Mixtral-8x7B-Instruct-v0.1"
        #"mistralai/Mixtral-8x22B-Instruct-v0.1"
	#"meta-llama/Llama-4-Scout-17B-16E"
	"Qwen/Qwen3-235B-A22B-FP8"
)

set -euo pipefail

platform="${1:-cuda}"

export NUM_GPUs=8

model="deepseek-ai/DeepSeek-R1-0528"
for tp_size in 4 8; do
	for batch_size in 16 32 64 128 -1; do
   		python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype fp8 --platform $platform
   		python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype fp8 -ep True --platform $platform
	done
done
