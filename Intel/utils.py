import argparse
import pandas as pd
import os
from vllm import LLM, SamplingParams
import torch

# Argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--tp", type=int, default=1, help="Gpus to be used in Tensor Parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Gpus to be used in Pipeline Parallelism")
    parser.add_argument("--ep", action='store_true', help="Expert Parallelism")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model")
    parser.add_argument("--power", action='store_true', help="Measure Power")
    return parser.parse_args()

def parse_arguments_single_run():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--tp", type=int, default=1, help="Gpus to be used in Tensor Parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Gpus to be used in Pipeline Parallelism")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallelism")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model")
    parser.add_argument("--power", action='store_true', help="Measure Power")
    parser.add_argument("--dtype", type=str, default='bfloat16')
    args = parser.parse_args()
    
    if args.batch_size == -1:
        args.batch_size = None

    return args

# Constent model loading between runs

def load_model(model_name, batch_size, dtype='bfloat16', tp = 1, pp = 1, ep = 1):
    """
    Load the model using vLLM. The GPUs are set to be used in tensor parallelism.
    """
    llm = LLM(
        model=model_name,
        max_num_seqs = batch_size,
        tokenizer=None,
        quantization=None,
        tensor_parallel_size = tp,
        pipeline_parallel_size=pp,
        data_parallel_size=ep,
        enable_expert_parallel= ep > 1,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=True,
        kv_cache_dtype='auto',
        device='xpu',
        block_size=16,
        enable_chunked_prefill=False, # not supported in xpu
        gpu_memory_utilization=0.90, # Less aggressive than default to avoid OOM
        load_format='auto',
        distributed_executor_backend='ray', # Necessary for vLLM to work with Intel XPU
        enable_prefix_caching=False,
        disable_sliding_window=False,
    )

    return llm

# Save results to CSV

def save_results(args, aggregated_data, path, batch_size = None):
    aggregated_data['Model Name'] = args.model_name
    aggregated_data['FrameWork'] = 'vLLM'
    aggregated_data['Hardware type'] = torch.xpu.get_device_name(torch.xpu.current_device()),
    aggregated_data['TP Size'] = args.tp
    aggregated_data['PP Size'] = args.pp
    aggregated_data['EP'] = args.ep
    if batch_size is not None:
        aggregated_data['Batch Size'] = batch_size
    else:
        aggregated_data['Batch Size'] = args.batch_size

    if os.path.exists(path):
        aggregated_data.to_csv(path, mode='a', index=False, header=False)
    else:
        aggregated_data.to_csv(path, mode='w', index=False, header=True)

def save_results_with_power(model_name, framework, hw, num, dtype, batch_size, in_len, out_len, ttft, latency, power_avg, active_power_avg, power_peak, active_power_peak, energy, active_energy):
    data = {
        'Model Name': [model_name],
        'FrameWork': [framework],
        'Hardware type': [hw],
        'Count': [num],
        'Precision': [dtype],
        'Batch Size': [batch_size],
        'In tokens': [in_len],
        'Out tokens': [out_len],
        'TTFT': [ttft * 1000],  # Convert TTFT to ms
        'Latency': [latency],
        'Power Avg': [power_avg],
        'Power Avg (Active)': [active_power_avg],
        'Power Peak': [power_peak],
        'Power Peak (Active)': [active_power_peak],
        'Energy': [energy],
        'Energy (Active)': [active_energy]
    }

    new_data_df = pd.DataFrame(data)
    file_path = '../Results/results.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_data_df], ignore_index=True)
    else:
        df = new_data_df

    df.to_csv(file_path, index=False)
