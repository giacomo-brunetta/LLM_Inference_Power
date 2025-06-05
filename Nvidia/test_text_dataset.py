from datasets import load_dataset
import time
import csv
import ray
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel
)
from profiler_utils import GPUProfiler, metrics
import torch
from utils import parse_arguments_single_run, load_model, save_results
from transformers import AutoTokenizer
import os

os.environ["VLLM_USE_V1"] = "0" # Per-request profiling is not in v1 engine, so use v0
def chat_len(chat):
    return sum([len(msg['content']) for msg in chat])

def processed_dataset(ds):
    """
    Load the dataset from Hugging Face and  process it for inference benchmarking.
    """

    inputs = []
    sampling_params = []

    for chat in ds['conversation']:
        input_chat = chat[:-1]
        output_chat = chat[-1]

        in_len_char = chat_len(input_chat)
        out_len_char = chat_len(output_chat)

        # Remove outliers that break models with maximum sequence length of 4096
        if in_len_char > 20000 or in_len_char < 64 or out_len_char > 20000 or out_len_char < 64:
            continue

        out_tokens = tokenizer(output_chat)["input_ids"]
        out_len = len(out_tokens)

        inputs.append(input_chat)

        sampling_params.append(SamplingParams(
                    n=1,
                    temperature= 1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens= out_len,
                ))
        
    print(f"Loaded {len(inputs)} samples")
    assert len(inputs) == len(sampling_params) and len(inputs) == 1000
    return inputs, sampling_params

args = parse_arguments_single_run()
model_name = args.model_name
interval = 0.5
batch_size = args.batch_size

print(f"Running {model_name}")
print(f"TP: {args.tp}")
print(f"PP: {args.pp}")
print(f"Batch size: {batch_size}")

# Load the dataset
ds = load_dataset("lmsys/lmsys-chat-1m", split=f"train[0:1483]") # 1000 samples after outlier removal
inputs, sampling_params = processed_dataset(ds)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load the model
torch._dynamo.config.suppress_errors = True

llm = load_model(model_name,batch_size, tp = args.tp, pp = args.pp, ep = args.ep, dtype=args.dtype)

# warm_up
print("Warming up...")
results = llm.chat(
    inputs[-50:],
    sampling_params=sampling_params[-50:],
    use_tqdm=True
)

print("Starting...")

# Create and start the profiler
gpu_profiler = GPUProfiler(gpus=os.environ['NUM_GPUs'], active_gpus=args.tp*args.pp)
gpu_profiler.start()

# Do inference
results = llm.chat(
    inputs,
    sampling_params=sampling_params,
    use_tqdm=True
)
gpu_profiler.stop() # Stop the profiler

# Profiler data to get latency, ttft, throughput and power metrics
latency_data, aggregated_data = metrics(results, gpu_profiler)

# Save the latency results in CSV files
latency_data.to_csv(f"../Results/Single_Runs/{model_name.split('/')[-1]}_tp{args.tp}_{args.pp}_bs{batch_size}.csv", index=True)
save_results(args, aggregated_data, '../Results/dataset_inference_results.csv')

ray.shutdown()
