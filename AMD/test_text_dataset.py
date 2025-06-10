from datasets import load_dataset, Dataset
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

os.environ["VLLM_USE_V1"] = "0"  # Per-request profiling is not in v1 engine, so use v0

def chat_len(chat):
    return sum([len(msg['content']) for msg in chat])

def processed_dataset(ds, tokenizer):
    inputs = []
    outputs = []

    for chat in ds["conversation"]:
        # Take everything except last message as "input", last message as "response"
        input_chat = chat[:-1]
        response = chat[-1]

        in_len = chat_len(input_chat)
        out_len = chat_len([response])

        # Only keep examples whose input/output lengths are between 64 and 20,000 chars
        if 64 <= in_len <= 20000 and 64 <= out_len <= 20000:
            inputs.append(input_chat)
            outputs.append(response)

    # Create a small Dataset from the filtered outputs
    outputs_ds = Dataset.from_list(outputs)

    # When batched=True, `batch["content"]` is a list of strings.
    def len_in_tokens(batch):
        enc = tokenizer(
            batch["content"],        # this is a list of strings
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        # enc["input_ids"] is now a list-of-lists, one per example in the batch.
        # We need to return a list of lengths, not just a single int.
        return {"out_len": [len(ids) for ids in enc["input_ids"]]}

    out_lengths_ds = outputs_ds.map(
        len_in_tokens,
        batched=True,
        batch_size=256,
        num_proc=64
    )

    out_lengths = out_lengths_ds["out_len"]

    sampling_params = [
        SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=ol,
        )
        for ol in out_lengths
    ]

    # Make sure we still have exactly 1,000 examples after filtering
    assert len(inputs) == len(sampling_params) and len(inputs) == 1000

    return inputs, sampling_params


def tokenize_message(message):
    encoding = tokenizer(
        message['content'],
        truncation=False,
        padding=False,
        return_attention_mask=True
    )
    input_ids = encoding["input_ids"]
    return {"out_len": len(input_ids)}


if __name__ == "__main__":
    args = parse_arguments_single_run()
    model_name = args.model_name
    interval = 0.5
    batch_size = args.batch_size

    print(f"Running {model_name}")
    print(f"TP: {args.tp}")
    print(f"PP: {args.pp}")
    print(f"Batch size: {batch_size}")

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:1483]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    inputs, sampling_params = processed_dataset(ds, tokenizer)

    # Load the model
    torch._dynamo.config.suppress_errors = True
    llm = load_model(model_name, batch_size, tp=args.tp, pp=args.pp, ep=args.ep, dtype=args.dtype)

    # Warm‐up
    print("Warming up...")
    _ = llm.chat(
        inputs[-50:],
        sampling_params=sampling_params[-50:],
        use_tqdm=True
    )

    print("Starting...")

    # Create and start the profiler
    gpu_profiler = GPUProfiler(gpus=int(os.environ['NUM_GPUs']), active_gpus=args.tp * args.pp)
    gpu_profiler.start()

    # Do inference
    results = llm.chat(
        inputs,
        sampling_params=sampling_params,
        use_tqdm=True
    )
    gpu_profiler.stop()  # Stop the profiler

    # Profiler data to get latency, ttft, throughput and power metrics
    latency_data, aggregated_data = metrics(results, gpu_profiler)

    # Save the latency results in CSV files
    #latency_data.to_csv(
    #    f"../Results/Single_Runs/{model_name.split('/')[-1]}_tp{args.tp}_{args.pp}_bs{batch_size}.csv",
    #    index=True
    #)
    save_results(args, aggregated_data, '../Results/dataset_inference_results.csv')

    ray.shutdown()

