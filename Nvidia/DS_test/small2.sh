models=(
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    "google/gemma-2-27b-it"
)

for tp in 2 4; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name "mistralai/Mistral-Small-3.1-24B-Instruct-2503" --tp $tp --batch_size $batch_size
    done
done


for tp in 1 2 4; do
    for batch_size in 16 32 64 128 -1; do
        python test_text_dataset.py --model_name "google/gemma-2-27b-it" --tp $tp --batch_size $batch_size
    done
done
