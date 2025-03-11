import argparse
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type for computation")
    parser.add_argument("--num_gpus", type=int, default=1, help="Gpus to be used")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model")
    parser.add_argument("--power", action='store_true', help="Measure Power")
    return parser.parse_args()

def parse_arguments_single_run():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--dtype", type=str, default="f16", help="Data type for computation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--in_len", type=int, default=128, help="In length")
    parser.add_argument("--out_len", type=int, default=128, help="Out length")
    parser.add_argument("--num_gpus", type=int, default=1, help="Gpus to be used")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model")
    parser.add_argument("--power", action='store_true', help="Measure Power")
    return parser.parse_args()

def save_results(model_name, framework, hw, num, dtype, batch_size, in_len, out_len, ttft, latency):
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
        'Latency': [latency]
    }

    new_data_df = pd.DataFrame(data)
    file_path = '../Results/results.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_data_df], ignore_index=True)
    else:
        df = new_data_df

    df.to_csv(file_path, index=False)

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