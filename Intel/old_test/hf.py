import torch
import time
from power_utils import power_profile_task
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
from utils import parse_arguments, save_results, save_results_with_power, print_args

args = parse_arguments()
print_args(args, multi_test=True)

# Map dtype argument to PyTorch dtype
dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16
}
dtype = dtype_map[args.dtype]

if args.dtype == "float16":
    print("Float16 might cause numerical instability")

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

# Move model to XPU first
device = "xpu"
model = model.to(device)

total_xpus = 4 # hardcoded, might fix in the future
active_tiles = args.num_gpus

if active_tiles > 1: # wrap with DataParallel
    device_ids = list(range(active_tiles))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    print("Using Dataparallel")

# Apply Intel IPEX optimization AFTER DataParallel
model = ipex.optimize(model, dtype=dtype)

# Set model to evaluation mode
model.eval()
model.to(device)

for batch_size in [1, 2, 4, 8, 16]:
    for len in [128, 256, 512, 1024, 2048]:

        in_len = len
        out_len = len

        input_ids = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, in_len))
        input_ids = input_ids.to(device)

        with torch.no_grad():
            with torch.amp.autocast('xpu', enabled=True, dtype=dtype):
                print("Warming up ...")
                # Warm-up
                output = model(input_ids)
                print("Latency Test...")
                # Measure Time to First Token (TTFT)
                torch.xpu.synchronize()
                start_time = time.perf_counter()
                output = model(input_ids)
                torch.xpu.synchronize()
                ttft = (time.perf_counter() - start_time)

                if args.power:
                    print("Measuring Power")
                    def f():
                        # Generate
                        generate_ids = model.generate(input_ids,
                                            max_new_tokens=out_len,
                                            num_beams=1,
                                            early_stopping=False)
                    
                    latency, power_avgs, power_peaks, energies = power_profile_task(f, 60, total_xpus)

                    active_power_average = sum([power_avgs[i] for i in range(active_tiles)])
                    total_power_average = sum(power_avgs)

                    active_power_peak = sum([power_peaks[i] for i in range(active_tiles)])
                    total_power_peak = sum(power_peaks)

                    active_energy = sum([energies[i] for i in range(active_tiles)])
                    total_energy = sum(energies)

                    save_results_with_power(
                        model_name,
                        'Transformers',
                        torch.cuda.get_device_name(torch.cuda.current_device()),
                        active_tiles,
                        dtype,
                        batch_size,
                        in_len,
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
                    # Measure time to generate new tokens
                    torch.xpu.synchronize()
                    start_time = time.perf_counter()
                    # Generate
                    generate_ids = model.generate(input_ids,
                                                max_new_tokens=out_len,
                                                num_beams=1,
                                                early_stopping=False)
                    torch.xpu.synchronize()
                    latency = (time.perf_counter() - start_time)

                    # Print results
                    print("\nResults:")
                    print(f"Latency: {latency:.3f} s")
                    print(f"Average TTFT: {ttft*1000:.3f} ms")

                    save_results(model_name,
                                'Transformers',
                                torch.xpu.get_device_name(torch.xpu.current_device()),
                                active_tiles,
                                dtype,
                                batch_size,
                                in_len,
                                out_len,
                                ttft,
                                latency)