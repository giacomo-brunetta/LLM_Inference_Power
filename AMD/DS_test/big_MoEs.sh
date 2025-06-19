models=(
	#"Qwen/Qwen3-235B-A22B-FP8"
	"deepseek-ai/DeepSeek-R1-0528"	
)


for model in "${models[@]}"; do
    for tp_size in 8 4 2; do
       for batch_size in -1; do
           python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8
       	   python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8 -ep True
   	done
   done
done
