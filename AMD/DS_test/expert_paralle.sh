for model in "${models[@]}"; do
    for tp_size in 8; do
       for batch_size in 16 32 64 128 -1; do
           python test_text_dataset.py --model_name "meta-llama/Llama-4-Scout-17B-16E-Instruct" -tp $tp_size --batch_size $batch_size -ep True
       done
   done
done

python test_text_dataset.py --model_name "mistralai/Mixtral-8x22B-Instruct-v0.1" -tp 4 --batch_size -1 -ep True

