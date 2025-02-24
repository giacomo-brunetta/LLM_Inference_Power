import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_utils import power_profile_task
from utils import parse_arguments, save_results

args = parse_arguments()

# Map dtype argument to PyTorch dtype
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}
dtype = dtype_map[args.dtype]

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
model.eval()

# generate random input according to batch size and in length
input_ids = torch.randint(low=0, high=model.config.vocab_size, size=(args.batch_size, args.in_len))

# Determine the number of available CUDA devices
num_devices = args.num_gpus

if num_devices > 1: # wrap with DataParallel
    device = torch.device('cuda')
    device_ids = list(range(num_devices))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    print("Using Dataparallel")
else:
    device = torch.device('cuda:0')

model.to(device)
input_ids = input_ids.to(device)

assert input_ids.shape[0] == args.batch_size and input_ids.shape[1] == args.in_len

with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
        print("Warm up...")
        output = model(input_ids) # Generate one token

        # Measure Time to First Token (TTFT)
        print("Measuring TTFT")
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        output = model(input_ids) # Generate one token

        torch.cuda.synchronize()
        ttft = (time.perf_counter() - start_time)  

        # Measure time to generate out_len new tokens
        print("Measuring Throughput")
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # generate one sequence
        # TODO does not work for DataParallel
        generate_ids = model.generate(input_ids,
                                        max_new_tokens=args.out_len,
                                        num_beams=1,
                                        early_stopping=False)

        torch.cuda.synchronize()
        latency = (time.perf_counter() - start_time)

        if args.power:
            print("Measuring Power")
            def f():
                return model(input_ids)
            
            power_profile_task(f, 60, 0.1)

        # Print results
        print("\n---------------Results----------------------")
        print(f"Latency: {latency:.3f} s")
        print(f"TTFT: {ttft*1000:.3f} ms")

save_results(model_name,
             'Transformers',
             torch.cuda.get_device_name(torch.cuda.current_device()),
             args.num_gpus,
             dtype,
             args.batch_size,
             args.in_len,
             args.out_len,
             ttft,
             latency)