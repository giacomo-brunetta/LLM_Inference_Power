echo "Four GPUs"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --gpu_ids=0,1,2,3  test_accelerate.py --model_name "meta-llama/Meta-Llama-3-8B" --num_gpus 4 --power
