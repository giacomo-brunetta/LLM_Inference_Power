models=(
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)

for model in "${models[@]}"; do
    for tp_size in 2 4; do
       for batch_size in 16 32 64 128 -1; do
           python test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -ep True
       done
   done
done
