models=(
    #"Qwen/Qwen2-72B-Instruct"
    #"mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-3.3-70B-Instruct"
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

for model in "${models[@]}"; do
    for tp_size in 4; do
       for batch_size in 32 64 128 -1; do
           python test_text_dataset.py --model_name $model --tp $tp_size --batch_size $batch_size
       done 
   done
done
