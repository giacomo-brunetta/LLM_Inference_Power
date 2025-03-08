"""
This script performs a single test with the model
"""

import time
from vllm import LLM, SamplingParams
import numpy as np
from utils import parse_arguments, save_results_with_power,save_results, print_args
from power_utils import power_profile_task
import torch

# Parse command-line arguments
args = parse_arguments()
print_args(args, multitest = False)
torch._dynamo.config.suppress_errors = True

total_gpus = torch.cuda.device_count()
active_gpus = args.num_gpus

if active_gpus > total_gpus:
    print(f"Unsupported Tensor Parallel Size {active_gpus}")
    exit(1)

if args.dtype == "default":
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=active_gpus,
        trust_remote_code=True,
        device='cuda',
    )
else:
    try:
        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=active_gpus,
            trust_remote_code=True,
            dtype=args.dtype,
            device='cuda',
        )
    except:
        print(f"Unsupported Data Type: {args.dtype}")
        exit(1)

batch_size = args.batch_size
input_len = args.in_len
out_len = args.out_len

sequence = SamplingParams(
    n=args.batch_size,
    temperature= 1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=out_len,
)

first_token = SamplingParams(
    n=args.batch_size,
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

if args.power:
    def f():
        llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)

    sampling_frequency = 0.5 #seconds

    latency, power_avgs, power_peaks, energies = power_profile_task(f, sampling_frequency, create_plot=False)

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
        args.dtype,
        args.batch_size,
        args.in_len,
        args.out_len,
        ttft,
        latency,
        total_power_average,
        active_power_average,
        total_power_peak,
        active_power_peak,
        total_energy,
        active_energy
    )
else:
    start_time = time.perf_counter()
    llm.generate(dummy_inputs, sampling_params=sequence, use_tqdm=False)
    end_time = time.perf_counter()
    latency = end_time - start_time

    save_results(
                args.model_name,
                'vLLM',
                torch.cuda.get_device_name(torch.cuda.current_device()),
                active_gpus,
                args.dtype,
                args.batch_size,
                args.in_len,
                args.out_len,
                ttft,
                latency)