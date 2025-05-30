import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import numpy as np
import matplotlib.pyplot as plt

from LLM_Inference_Power.Nvidia.profiler_utils import power_profile_task, gpuPowerProbe
from utils import parse_arguments, save_results, save_results_with_power

args = parse_arguments()

create_plot = False

# Map dtype argument to PyTorch dtype
dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

dtype = dtype_map[args.dtype]

total_gpus = 4 #torch.cuda.device_count()
active_gpus = args.num_gpus

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision=args.dtype)
device = accelerator.device

# Load tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
)

# Prepare model and dataloader with accelerator
model = accelerator.prepare(model)

# Set model to evaluation mode
model.eval()

# Set pad_token to eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    for lenght in [128, 256, 512, 1024, 2048]:

        print(f"Batch size: {batch_size}, Lenght: {lenght}, total_gpus: {total_gpus}, active_gpus: {active_gpus}")

        input_len = lenght
        out_len = lenght

        input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(batch_size,input_len))

        input_ids = input_ids.to(accelerator.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=dtype):

                inference_model = accelerator.unwrap_model(model)

                if accelerator.is_main_process:
                    print("Warm up...")
                output = inference_model(input_ids) # Generate one token

                # Measure Time to First Token (TTFT)
                if accelerator.is_main_process:
                    print("Measuring TTFT")
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                output = inference_model(input_ids) # Generate one token

                if accelerator.is_main_process:
                    torch.cuda.synchronize()
                    ttft = (time.perf_counter() - start_time)  

                # Measure time to generate out_len new token
                if accelerator.is_main_process:
                    print("Measuring Throughput")

                sampling_frequency = 0.5 #seconds

                if args.power:
                    if accelerator.is_main_process:
                        inference_powers = []
                        inference_powers_time = []
                        power_avgs = []
                        power_peaks = []
                        energies = []
                        power_profiles = []

                        for id in range(total_gpus):
                            power_profiles.append(gpuPowerProbe(interval=sampling_frequency, gpu_id=id))
                            power_profiles[id].start()

                        start_time = time.perf_counter()
                    
                    generate_ids = inference_model.generate(input_ids,
                                            max_new_tokens=out_len,
                                            num_beams=1,
                                            early_stopping=False)

                    if accelerator.is_main_process:
                        end_time = time.perf_counter()

                        latency = end_time - start_time

                        for power_profile in power_profiles:
                            power, times = power_profile.stop()
                            inference_powers.append(power)
                            inference_powers_time.append(times)
                            power_profile.destroy()

                        print("\n----------------Power-----------------------")
                        for id in range(total_gpus):
                            print(f"GPU {id}:")
                            power = np.array(inference_powers[id]) / 1000  # to Watt
                            times = np.array(inference_powers_time[id])
                            avg_power = np.mean(power)
                            peak_power = np.max(power)
                            energy = np.sum(power*times)

                            power_avgs.append(avg_power)
                            power_peaks.append(peak_power)
                            energies.append(energy)

                            print(f"    Power avg : {avg_power :.3f} W")
                            print(f"    Power peak: {peak_power :.3f} W")
                            print(f"    Energy    : {energy :.3f} J")
                            if create_plot:
                                plt.plot(np.cumsum(inference_powers_time[id]), power, label=f'GPU {id}')

                        if create_plot:
                                plt.title("Power consumption")
                                plt.legend()
                                plt.xlabel(f'Time ({sampling_frequency} sec intervals)')
                                plt.ylabel('Power Consumption (W)')
                                plt.savefig('gpu_power_plot.png')

                        active_power_average = sum([power_avgs[i] for i in range(active_gpus)])
                        total_power_average = sum(power_avgs)

                        active_power_peak = sum([power_peaks[i] for i in range(active_gpus)])
                        total_power_peak = sum(power_peaks)

                        active_energy = sum([energies[i] for i in range(active_gpus)])
                        total_energy = sum(energies)

                        print("Saving Results")

                        save_results_with_power(
                            args.model_name,
                            'Accelerate',
                            torch.cuda.get_device_name(torch.cuda.current_device()),
                            active_gpus,
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

                    generate_ids = accelerator.unwrap_model(model).generate(input_ids,
                                            max_new_tokens=out_len,
                                            num_beams=1,
                                            early_stopping=False)
                    
                    end_time = time.perf_counter()

                    latency = end_time - start_time
                    save_results(
                                args.model_name,
                                'Accelerate',
                                torch.cuda.get_device_name(torch.cuda.current_device()),
                                active_gpus,
                                dtype,
                                batch_size,
                                input_len,
                                out_len,
                                ttft,
                                latency)

accelerator.free_memory()