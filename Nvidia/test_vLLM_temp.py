"""
This python scripts loads the model once and then performs thest for multiple batch sizes and sequence lenghts
Recommended when a comprehensive test is being run, since model loading often takes more than inference.
"""

import time
from vllm import LLM, SamplingParams
import numpy as np
from utils import parse_arguments, save_results_with_power, save_results
from power_utils import power_profile_task
import torch
import traceback
import sys

# Parse command-line arguments
args = parse_arguments()

total_gpus = torch.cuda.device_count()
active_gpus = args.num_gpus

if active_gpus > total_gpus:
    print(f"Unsupported Tensor Parallel Size {active_gpus}")
    exit(1)

try:
    llm = LLM(
        model=args.model_name,
        speculative_model=None,
        num_speculative_tokens=None,
        speculative_draft_tensor_parallel_size=None,
        tokenizer=None,
        quantization=None,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=4096 ,
        enforce_eager=True,
        kv_cache_dtype='auto',
        device='cuda',
        block_size=16,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.99,
        load_format='auto',
        distributed_executor_backend=None,
        enable_prefix_caching=False,
        disable_sliding_window=True,
    )
except Exception as e:
    print(f"An error occurred while loading model {args.model_name}:", file=sys.stderr)
    print(f"Error type: {type(e).__name__}", file=sys.stderr)
    print(f"Error message: {e}", file=sys.stderr)
    # Optional: Print the full traceback for more detailed debugging
    # traceback.print_exc(file=sys.stderr)
    exit(1)

# Modify the batch sizes and sequence lenght here if needed

for batch_size in [32, 64]:
    for length in [128, 256, 512, 1024, 2048]:

        input_len = length
        out_len = length

        print("\n----------------Testing----------------\n")
        print(f"In lenght: {input_len}")
        print(f"Out Lenght: {out_len}")
        print(f"Batch size: {batch_size}")

        sequence = SamplingParams(
            n=1,
            temperature= 1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=out_len,
        )

        first_token = SamplingParams(
            n=1,
            max_tokens=1,
        )

        dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size,input_len))

        dummy_inputs = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        print("Warming up...")
        
        start_time = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)
        end_time = time.perf_counter()
        print("Warmup time: ", end_time - start_time)

        dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size,input_len))

        dummy_inputs = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        print("Measurng TTFT")
        start_time = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)
        end_time = time.perf_counter()
        ttft = end_time - start_time

        print("TTFT: ", ttft)

        print("Measuring Latency")

        def f():
            llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)

        sampling_frequency = 0.5 #seconds

        latency, power_avgs, power_peaks, energies = power_profile_task(f, sampling_frequency, active_gpus, total_gpus, create_plot=False)

        print("Latency: ", latency)

        active_power_average = sum([power_avgs[i] for i in range(active_gpus)])
        total_power_average = sum(power_avgs)

        active_power_peak = sum([power_peaks[i] for i in range(active_gpus)])
        total_power_peak = sum(power_peaks)

        active_energy = sum([energies[i] for i in range(active_gpus)])
        total_energy = sum(energies)

        save_results_with_power(
            args.model_name,
            'vLLM',
            torch.cuda.get_device_name(torch.cuda.current_device()),
            active_gpus,
            'bfloat16',
            batch_size,
            input_len,
            out_len,
            ttft,
            latency,
            total_power_average,
            active_power_average,
            total_power_peak,
            active_power_peak,
            total_energy,
            active_energy
        )