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

# Parse command-line arguments
args = parse_arguments()

torch._dynamo.config.suppress_errors = True

total_gpus = torch.xpu.device_count()
active_gpus = args.num_gpus

if active_gpus > total_gpus:
    print(f"Unsupported Tensor Parallel Size {active_gpus}")
    exit(1)

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

dtype = dtype_map[args.dtype]

try:
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        dtype=dtype,
        device='xpu',
    )
except:
    print(f"Unsupported Data Type: {args.dtype}")
    exit(1)

# Modify the batch sizes and sequence lenght here if needed

for batch_size in [8]:
    for len in [128, 256, 512]:

        if len >= 512 and batch_size > 8:
            continue

        input_len = len
        out_len = len

        print("Testing")
        print(f"In lenght: {input_len}")
        print(f"Out Lenght: {out_len}")
        print(f"Batch size: {batch_size}")

        sequence = SamplingParams(
            n=batch_size,
            temperature= 1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=out_len,
        )

        first_token = SamplingParams(
            n=batch_size,
            max_tokens=1,
        )

        dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size,input_len))

        dummy_inputs = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]

        print("Warming up...")
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)

        print("Measurng TTFT")
        start_time = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)
        end_time = time.perf_counter()
        ttft = end_time - start_time

        print("Measuring Latency")
        start_time = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time

        print(f"Latency: {latency}")
        print(f"TTFT: {ttft}")

        def f():
            llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)

        latency, power_avgs, power_peaks, energies = power_profile_task(f, 30, 7)

        active_power_average = sum([power_avgs[i] for i in range(active_gpus)])
        total_power_average = sum(power_avgs)

        active_power_peak = sum([power_peaks[i] for i in range(active_gpus)])
        total_power_peak = sum(power_peaks)

        active_energy = sum([energies[i] for i in range(active_gpus)])
        total_energy = sum(energies)

        save_results_with_power(
            args.model_name,
            'vLLM',
            torch.xpu.get_device_name(torch.xpu.current_device()),
            active_gpus,
            args.dtype,
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