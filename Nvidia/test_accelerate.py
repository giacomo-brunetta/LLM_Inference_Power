import os
import time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model, infer_auto_device_map
import numpy as np
import matplotlib.pyplot as plt

from power_utils import power_profile_task, gpuPowerProbe
from utils import parse_arguments, save_results, save_results_with_power

args = parse_arguments()

create_plot = False

# Initialize distributed environment
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Map dtype argument to PyTorch dtype
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

dtype = dtype_map[args.dtype]

total_gpus = torch.cuda.device_count()
active_gpus = args.num_gpus

try:
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token to eos_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype)

    # Accelerate optimal allocation
    device_map = infer_auto_device_map(model)
    model = dispatch_model(model, device_map=device_map)

    model.eval()

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        for len in [128, 256, 512, 1024, 2048]:
            input_len = len
            out_len = len

            # generate random input according to batch size and in length
            input_ids = torch.randint(low=0, high=model.config.vocab_size, size=(args.batch_size, args.in_len))

            embedding_device = next(model.parameters()).device
            input_ids = input_ids.to(embedding_device)

            assert input_ids.shape[0] == args.batch_size and input_ids.shape[1] == args.in_len

            if rank == 0:
                print("Testing")
                print(f"In lenght: {input_len}")
                print(f"Out Lenght: {out_len}")
                print(f"Batch size: {batch_size}")

            with torch.no_grad():
                with torch.amp.autocast(device_type=embedding_device.type, dtype=dtype):
                    if rank == 0:
                        print("Warm up...")
                    output = model(input_ids) # Generate one token

                    # Measure Time to First Token (TTFT)
                    if rank == 0:
                        print("Measuring TTFT")
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    output = model(input_ids) # Generate one token

                    torch.cuda.synchronize()
                    ttft = (time.perf_counter() - start_time)  

                    # Measure time to generate out_len new tokens
                    if rank == 0:
                        print("Measuring Throughput")

                    sampling_frequency = 0.5 #seconds

                    if args.power:
                        if rank == 0:
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
                        
                        generate_ids = model.generate(input_ids,
                                                max_new_tokens=args.out_len,
                                                num_beams=1,
                                                early_stopping=False)

                        end_time = time.perf_counter()

                        latency = end_time - start_time

                        if rank == 0:
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

                                save_results_with_power(
                                    args.model_name,
                                    'Transformers',
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

                        generate_ids = model.generate(input_ids,
                            max_new_tokens=args.out_len,
                            num_beams=1,
                            early_stopping=False)
                        
                        end_time = time.perf_counter()

                        latency = end_time - start_time
                        if rank == 0:
                            save_results(
                                        args.model_name,
                                        'Transformers',
                                        torch.cuda.get_device_name(torch.cuda.current_device()),
                                        active_gpus,
                                        args.dtype,
                                        args.batch_size,
                                        args.in_len,
                                        args.out_len,
                                        ttft,
                                        latency)

finally:
    # Ensure cleanup happens
    dist.destroy_process_group()