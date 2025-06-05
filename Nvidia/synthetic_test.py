"""
This python scripts loads the model once and then performs thest for multiple batch sizes and sequence lenghts
Recommended when a comprehensive test is being run, since model loading often takes more than inference.
"""

import time
from vllm import LLM, SamplingParams
import numpy as np
from utils import parse_arguments, load_model, save_results
from profiler_utils import GPUProfiler, metrics
import torch
import traceback
import sys

import os
# os.environ["VLLM_USE_V1"] = "1"

# Parse command-line arguments
args = parse_arguments()

total_gpus = torch.cuda.device_count()
active_gpus = args.pp * args.tp

try:
    llm = load_model(args.model_name, tp = args.tp, pp=args.pp, batch_size=1)
except Exception as e:
    print(f"An error occurred while loading model {args.model_name}:", file=sys.stderr)
    print(f"Error type: {type(e).__name__}", file=sys.stderr)
    print(f"Error message: {e}", file=sys.stderr)
    exit(1)

# Modify the batch sizes and sequence lenght here if needed

print("Warmup")

sampling_params = SamplingParams(
    n=1,
    temperature= 1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=10,
)
dummy_prompt_token_ids = np.random.randint(10000, size=(1,128))
dummy_inputs = [{
    "prompt_token_ids": batch
} for batch in dummy_prompt_token_ids.tolist()]

llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)


# Begin the test
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
    for length in [128, 256, 512, 1024, 2048]:

        input_len = length
        out_len = length

        print("\n----------------Testing----------------\n")
        print(f"In lenght: {input_len}")
        print(f"Out Lenght: {out_len}")
        print(f"Batch size: {batch_size}")

        sampling_params = SamplingParams(
            n=1,
            temperature= 1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=out_len,
        )

        dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size,input_len))

        dummy_inputs = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        print("Measuring Latency")
        power_profiler = GPUProfiler(gpus=torch.cuda.device_count(), active_gpus=args.pp*args.tp)

        power_profiler.start()

        results = llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)

        power_profiler.stop()

        sampling_frequency = 0.5 #seconds

        # latency_data, aggregated_data = metrics(results, power_profiler, verbose=True)

        # save_results(args, aggregated_data, '../Results/syntethic_inference_results.csv', batch_size=batch_size)
