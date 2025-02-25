import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
from utils import parse_arguments, save_results

args = parse_arguments()

# Map dtype argument to PyTorch dtype
dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16
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

# Determine the number of available XPU devices
num_devices = args.num_gpus

if num_devices > 1: # wrap with DataParallel
    device_ids = list(range(num_devices))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    print("Using Dataparallel")

# Apply Intel IPEX optimization AFTER DataParallel
model = ipex.optimize(model, dtype=dtype)

# Set model to evaluation mode
model.eval()
model.to(device)

input_ids = torch.randint(low=0, high=model.config.vocab_size, size=(args.batch_size, args.in_len))
input_ids = input_ids.to(device)

assert input_ids.shape[0] == args.batch_size and input_ids.shape[1] == args.in_len

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

        # Measure time to generate new tokens
        torch.xpu.synchronize()
        start_time = time.perf_counter()
        # Generate
        generate_ids = model.generate(input_ids,
                                    max_new_tokens=args.out_len,
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
             args.num_gpus,
             dtype,
             args.batch_size,
             args.in_len,
             args.out_len,
             ttft,
             latency)