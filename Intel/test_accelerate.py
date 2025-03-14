import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import intel_extension_for_pytorch as ipex
import numpy as np
import matplotlib.pyplot as plt
import threading

from power_utils import power_profile_task
from utils import parse_arguments, save_results, save_results_with_power

args = parse_arguments()

create_plot = False

# Map dtype argument to PyTorch dtype
dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

dtype = dtype_map[args.dtype]

total_xpus = torch.xpu.device_count()
active_xpus = args.num_gpus

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision=args.dtype)
device = accelerator.device

# Load tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, mixed_precision=args.dtype)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
)

# Prepare model and dataloader with accelerator
model = ipex.optimize(model, dtype=dtype)
model = accelerator.prepare(model)
model.eval()


# Set pad_token to eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    for lenght in [128, 256, 512, 1024, 2048]:

        print(f"Batch size: {batch_size}, Lenght: {lenght}, total_gpus: {total_xpus}, active_gpus: {active_xpus}")
        
        # Skip tests that are too big
        if lenght >= 512 and batch_size > 8:
            continue

        input_len = lenght
        out_len = lenght

        input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(batch_size,input_len))

        input_ids = input_ids.to(accelerator.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    
                    inference_model = accelerator.unwrap_model(model)

                    print("Warm up...")
                    output = inference_model(input_ids) # Generate one token

                    # Measure Time to First Token (TTFT)
                    print("Measuring TTFT")
                    torch.xpu.synchronize()
                    start_time = time.perf_counter()

                    output = inference_model(input_ids) # Generate one token

                    torch.xpu.synchronize()
                    ttft = (time.perf_counter() - start_time)  

                    # Measure time to generate out_len new token
                    print("Measuring Throughput")

                    if args.power:
                        inference_powers = []
                        inference_powers_time = []
                        power_avgs = []
                        power_peaks = []
                        energies = []
                        power_profiles = []

                        power_probing_threads = []
                        stop = threading.Event()  # Create the stopping event

                        # Create and start power probing threads
                        for xpu in range(total_xpus):
                            tile1_pow = []
                            tile2_pow = []
                            intervals = []
                            inference_powers.append((tile1_pow, tile2_pow))
                            inference_powers_time.append([intervals])
                            
                            thread = threading.Thread(target=power_probe, args=(xpu, stop, tile1_pow, tile2_pow, intervals))
                            power_probing_threads.append(thread)
                            thread.start()

                        start_time = time.perf_counter()
                        generate_ids = inference_model.generate(input_ids,
                                                max_new_tokens=out_len,
                                                num_beams=1,
                                                early_stopping=False)
                        latency = time.perf_counter() - start_time

                        # Repeat if the job took less than 30s
                        while time.perf_counter() - start_time < 30: # 30s
                            generate_ids = accelerator.unwrap_model(model).generate(input_ids,
                                                max_new_tokens=out_len,
                                                num_beams=1,
                                                early_stopping=False)

                        stop.set()  # Stop power probing threads
                        
                        for thread in power_probing_threads:
                            thread.join()

                        power_avgs = []
                        power_peaks = []
                        energies = []

                        print("\n----------------Power-----------------------")
                        for id in range(total_xpus):

                            print(f"GPU {id}:")
                            power1 = np.array(inference_powers[id*2])
                            power2 = np.array(inference_powers[id*2 +1])
                            times = np.array(inference_powers_time[id])

                            avg_power1 = np.mean(power1)
                            peak_power1 = np.max(power1)
                            energy1 = np.sum(power1*times)

                            avg_power2 = np.mean(power2)
                            peak_power2 = np.max(power2)
                            energy2 = np.sum(power2*times)

                            power_avgs.append(avg_power1, avg_power2)
                            power_peaks.append(peak_power1, peak_power2)
                            energies.append(energy1, energy2)

                            print("  Tile 0:")
                            print(f"    Power avg : {avg_power1 :.3f} W")
                            print(f"    Power peak: {peak_power1 :.3f} W")
                            print(f"    Energy    : {energy1 :.3f} J")

                            print("  Tile 1:")
                            print(f"    Power avg : {avg_power2 :.3f} W")
                            print(f"    Power peak: {peak_power2 :.3f} W")
                            print(f"    Energy    : {energy2 :.3f} J")

                        if create_plot:
                            plt.title("Power consumption")
                            plt.legend()
                            plt.xlabel(f'Time (sec)')
                            plt.ylabel('Power Consumption (W)')
                            plt.savefig('gpu_power_plot.png')

                        active_power_average = sum([power_avgs[i] for i in range(active_xpus)])
                        total_power_average = sum(power_avgs)

                        active_power_peak = sum([power_peaks[i] for i in range(active_xpus)])
                        total_power_peak = sum(power_peaks)

                        active_energy = sum([energies[i] for i in range(active_xpus)])
                        total_energy = sum(energies)

                        print("Saving Results")

                        save_results_with_power(
                            args.model_name,
                            'Accelerate',
                            torch.xpu.get_device_name(torch.xpu.current_device()),
                            active_xpus,
                            dtype,
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
                    else:
                        start_time = time.perf_counter()

                        generate_ids = inference_model.generate(
                                input_ids,
                                max_new_tokens=out_len,
                                num_beams=1,
                                early_stopping=False)
                        
                        end_time = time.perf_counter()

                        latency = end_time - start_time
                        save_results(
                                args.model_name,
                                'Accelerate',
                                torch.xpu.get_device_name(torch.xpu.current_device()),
                                active_xpus,
                                dtype,
                                batch_size,
                                input_len,
                                out_len,
                                ttft,
                                latency) 