import time
from vllm import LLM, SamplingParams
import numpy as np
import pandas as pd
import os
import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run inference.")
parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float16", help="Data type for computation")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--in_len", type=int, default=512, help="In length")
parser.add_argument("--out_len", type=int, default=512, help="Out length")
parser.add_argument("--model_name", type=str, default = "meta-llama/Llama-2-7b-hf", help="Model")
args = parser.parse_args()

llm = LLM(
    model=args.model_name,
    tensor_parallel_size=args.batch_size,
    trust_remote_code=True,
    dtype=args.dtype,
    device='cuda'
)

batch_size = args.batch_size
input_len = args.in_len
out_len = args.out_len

sampling_params = SamplingParams(
    n=1,
    temperature= 1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=out_len,
)
 
print(sampling_params)
dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size,input_len))

dummy_inputs = [{
    "prompt_token_ids": batch
} for batch in dummy_prompt_token_ids.tolist()]

print("Warming up...")
llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)

start_time = time.perf_counter()
llm.generate(dummy_inputs, sampling_params=sampling_params, use_tqdm=False)
end_time = time.perf_counter()
latency = end_time - start_time

total_tokens = batch_size*(input_len + out_len)

print("\nResults:")
print(f"Latency: {latency:.3f} s")

data = {
    'Model Name': [args.model_name],
    'FrameWork': ['vLLM'],
    'Precision': [args.dtype],
    'Batch Size': [args.batch_size],
    'In tok': [args.in_len],
    'Out tok': [args.out_len],
    'TTFT (ms)': ['N/A'],  # Convert TTFT to ms
    'Latency' : [latency]
}

# Create the DataFrame
new_data_df = pd.DataFrame(data)

file_path = 'results.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df = pd.concat([df, new_data_df], ignore_index=True)
else:
    df = new_data_df

df.to_csv(file_path, index=False)