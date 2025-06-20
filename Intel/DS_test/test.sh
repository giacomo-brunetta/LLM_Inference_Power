export NUM_GPUs=8

models=(
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${models[@]}"; do
    for tp in 8 16; do
        for batch_size in -1; do
            python3 test_text_dataset.py --model_name $model --tp $tp --batch_size $batch_size
        done
    done
done

