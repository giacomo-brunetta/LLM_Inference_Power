
for batch_size in -1 16 32 64 128; do
        python test_text_dataset.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --ep --tp 4 --batch_size $batch_size
done
