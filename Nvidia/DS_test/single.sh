
for batch_size in -1 16 32 64 128; do
        python test_text_dataset.py --model_name meta-llama/Llama-3.1-8B-Instruct --tp 4 --batch_size $batch_size
done
