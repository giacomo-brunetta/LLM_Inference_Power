models=(
	Qwen/Qwen3-235B-A22B
	)


for model in "${models[@]}"; do
    for tp_size in 4; do
       for batch_size in 16 32 64 128 -1; do
	   python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8
           python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8 -ep True
       done
   done
done
