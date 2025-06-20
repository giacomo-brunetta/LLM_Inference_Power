"""
This python scripts loads the model once and then performs thest for multiple batch sizes and sequence lenghts
Recommended when a comprehensive test is being run, since model loading often takes more than inference.
"""

import ray
import json
import time
from vllm import LLM, SamplingParams
import numpy as np
from utils import parse_arguments, save_results_with_power, save_results
from power_utils import power_profile_task
import torch

# Parse command-line arguments
args = parse_arguments()

# torch._dynamo.config.suppress_errors = True

total_gpus = torch.xpu.device_count()
active_gpus = args.num_gpus

ray.init(
    num_cpus=32,
    num_gpus=active_gpus,
    object_store_memory=32 * 1024**3,  # 32 GB
    _system_config={
        "object_spilling_config": json.dumps({
            "type": "filesystem",
            "params": {
                "directory_path": "/root/.cache/ray_tmp/spill"
            }
        })
    }
)

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
        enforce_eager=False,
        kv_cache_dtype='auto',
        device='xpu',
        block_size=16,
        gpu_memory_utilization=0.9,
        load_format='auto',
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        disable_sliding_window=True,
        distributed_executor_backend='ray',
    )
except:
    print(f"Unsupported Data Type: {args.dtype}")
    exit(1)

# Modify the batch sizes and sequence lenght here if needed

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    for length in [128, 256, 512, 1024, 2048]:

        input_len = length
        out_len = length

        print("\n----------------Testing----------------\n")
        print(f"In lenght : {input_len}")
        print(f"Out lenght: {out_len}")
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
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)

        print("Measurng TTFT")
        start_time = time.perf_counter()
        llm.generate(dummy_inputs, sampling_params=first_token, use_tqdm=False)
        end_time = time.perf_counter()
        ttft = end_time - start_time

        print(f"TTFT: {ttft}")

        def f():
            llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)

        latency, power_avgs, power_peaks, energies = power_profile_task(f, 10, 4)

        print(f"Latency: {latency}")

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

ray.shutdown()