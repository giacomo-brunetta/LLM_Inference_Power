export NUM_GPUs=4

for tp_size in 2; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" --tp $tp_size --batch_size $batch_size
    done 
done


for tp_size in 2 4; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --tp $tp_size --batch_size $batch_size
    done 
done

for tp_size in 2 4; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name Qwen/Qwen2-72B-Instruct --tp $tp_size --batch_size $batch_size
    done 
done
