model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
platform='cuda'



for tp_size in 4; do
   for batch_size in 32 64 128 -1; do
       python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype compressed --platform $platform
       python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype compressed -ep True --platform $platform
    done
done


model="Qwen/Qwen3-235B-A22B-FP8"

for tp_size in 4; do
   for batch_size in 32 64 128 -1; do
       python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype fp8 --platform $platform
       python3 test_text_dataset.py --model_name $model -tp $tp_size --batch_size $batch_size -dtype fp8 -ep True --platform $platform
    done
done
