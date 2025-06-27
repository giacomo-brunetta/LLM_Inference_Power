models=(
    "meta-llama/Llama-3.3-70B-Instruct"
    )


for model in "${models[@]}"; do
    for tp_size in 1 2 4; do
       for batch_size in 32 64 128 -1; do
           python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size --dtype fp8
       done
   done
done
