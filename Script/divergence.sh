platform="${1:-cuda}"

export NUM_GPUs=4

 models=(
	"meta-llama/Llama-3.1-8B-Instruct"
	"Qwen/Qwen3-8B"
	"mistralai/Ministral-8B-Instruct-2410"
	"Qwen/Qwen3-14B"
	"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
	"Qwen/Qwen3-32B"
	)

for model in "${models[@]}"; do
      #python3 KL_test.py --model_name_1  $model --model_name_2 $model
      python3 KL_test.py --model_name_1  $model --model_name_2 $model --data_type_2 fp8
done

models=(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    #"Qwen/Qwen2-72B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${models[@]}"; do
	#python3 KL_test.py --model_name_1  $model --model_name_2 $model -tp 2
	python3 KL_test.py --model_name_1  $model --model_name_2 $model --data_type_2 fp8 -tp 2
done
