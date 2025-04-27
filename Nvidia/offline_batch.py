import ray
import time
from datasets import load_dataset
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from ray.data import DataContext
from tqdm import tqdm
import numpy as np
import threading
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
import logging
import re
import os
import torch

logging.getLogger("ray").setLevel(logging.CRITICAL)

from power_utils import gpuPowerProbe

def sum_operator_durations(stats_str: str) -> float:
    """
    Sum all durations that appear as 'blocks produced in <seconds>s'
    inside a Ray Data stats() string.
    """
    matches = re.findall(r'blocks produced in ([0-9]*\.?[0-9]+)s', stats_str)
    return sum(map(float, matches)) if matches else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Processing Script")
    
    # Add arguments with default values
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name to use")
    parser.add_argument("--cpus", type=int, default=20,
                        help="Number of CPUs to use")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=250,
                        help="Maximum number of tokens for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Level of concurrency")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of rows to process")
    
    return parser.parse_args()

def count_chars(text):
    if isinstance(text, str):
        return len(text)
    return 0

def count_messages_chars(messages):
    total = 0
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            total += count_chars(msg["content"])
    return total

def print_chat(ds):
    for row in ds.take_all():
        print("Conversation:")
        for i, msg in enumerate(row["conversation"]):
            print(f"  Message {i+1}:")
            print(f"    Role: {msg['role']}")
            print(f"    Content: {msg['content']}") 
        print("-" * 80)

# Metrics tracking variables
class Metrics:
    def __init__(self, interval=1, gpus=1):
        # Metrics configuration
        self.started = False
        self.lock = threading.Lock()
        self.interval=interval
        self.gpus = gpus

        # Metrics tracking variables
        self.input_chars = 0
        self.output_chars = 0
        self.total_rows = 0
        self.start_time = None
        self.end_time = None

        self.total_time = None
        self.inference_time = None

        self.inference_powers = []
        self.inference_powers_time = []
        self.power_avgs = []
        self.power_peaks = []
        self.energies = []
        self.power_profiles = []

    def start(self):
        self.start = True
        self.start_time = time.time()
        for id in range(self.gpus):
            self.power_profiles.append(gpuPowerProbe(interval=self.interval, gpu_id=id))
            self.power_profiles[id].start()

    def stop(self, ds=None):
        self.end_time = time.time()

        self.total_time = self.end_time - self.start_time

        if ds != None:
            self.inference_time = sum_operator_durations(processed_ds.stats())
        else:
            self.inference_time = self.total_time

        for power_profile in self.power_profiles:
            power, times = power_profile.stop()
            self.inference_powers.append(power)
            self.inference_powers_time.append(times)
            power_profile.destroy()

        for id in range(self.gpus):
            print(f"GPU {id}:")
            power = np.array(self.inference_powers[id]) / 1000
            times = np.array(self.inference_powers_time[id])

            # If timing included warump or idle time, get rid of it
            if self.total_time > self.inference_time:

                samples_to_keep = int(self.inference_time / self.interval)

                power = power[-samples_to_keep:]
                times = times[-samples_to_keep:]

                assert samples_to_keep == len(power)
                assert len(power) == len(times)

            avg_power = np.mean(power)
            peak_power = np.max(power)
            energy = np.sum(power*times)

            self.power_avgs.append(avg_power)
            self.power_peaks.append(peak_power)
            self.energies.append(energy)
            
    
    def add_row(self, input_count, output_count):
        with self.lock:
            self.input_chars += input_count
            self.output_chars += output_count
            self.total_rows += 1
    
    def print_stats(self):
        input_tps = self.input_chars / self.inference_time if self.inference_time > 0 else 0
        output_tps = self.output_chars / self.inference_time if self.inference_time > 0 else 0
        combined_tps = (self.input_chars + self.output_chars) / self.inference_time if self.inference_time > 0 else 0
        
        print("\n===== THROUGHPUT STATS =====")
        print(f"Total elapsed time: {self.total_time:.2f} seconds")
        print(f"Total inference time: {self.inference_time:.2f} seconds")
        print(f"Loading time: {self.total_time - self.inference_time:.2f} seconds")
        print(f"Total rows processed: {self.total_rows}")
        print(f"Total rows per second: {self.total_rows / self.inference_time:.2f} rows/second")
        print(f"Total input chars: {self.input_chars}")
        print(f"Total output chars: {self.output_chars}")
        print(f"Input chars throughput: {input_tps:.2f} chars/second")
        print(f"Output chars throughput: {output_tps:.2f} chars/second")
        print(f"Combined throughput: {combined_tps:.2f} chars/second")

        print("\n===== POWER STATS =====")
        for id in range(self.gpus):
            print(f"GPU {id}:")
            print(f"    Power avg : {self.power_avgs[id] :.3f} W")
            print(f"    Power peak: {self.power_peaks[id] :.3f} W")
            print(f"    Energy    : {self.power_peaks[id] :.3f} J")

    def save_to_pandas(self, args=None):
        metrics_dict = {
            "model_name": [args.model_name if args else None],
            "device": [torch.cuda.get_device_name(torch.cuda.current_device())],
            "gpus": [args.gpus if args else None],
            "max_tokens": [args.max_tokens if args else None],
            "batch_size": [args.batch_size if args else None],
            "concurrency": [args.concurrency if args else None],
            "rows": [args.rows if args else None],
            "total_rows": [self.total_rows],
            "input_chars": [self.input_chars],
            "output_chars": [self.output_chars],
            "inference_time": [self.inference_time],
            "total_time": [self.total_time],
            "avg_power": [np.sum(self.power_avgs)],
            "total_energy": [np.sum(self.energies)],
        }

        new_data_df = pd.DataFrame(metrics_dict)
        file_path = '../Results/results_mlsys.csv'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.concat([df, new_data_df], ignore_index=True)
        else:
            df = new_data_df

        df.to_csv(file_path, index=False)

args = parse_args()

# Access arguments
print(f"Model: {args.model_name}")
print(f"CPUs: {args.cpus}")
print(f"GPUs: {args.gpus}")
print(f"Max tokens: {args.max_tokens}")
print(f"Batch Size: {args.batch_size}")
print(f"Concurrency: {args.concurrency}")
print(f"Rows: {args.rows}")

# Initialize metrics tracker
metrics = Metrics(gpus = args.gpus)

# Define a shorter temporary directory path
short_temp_dir = "/home/gbrun/tmp/ray"

# Initialize Ray with the specified temporary directory
ray.init(
    num_cpus=args.cpus,
    num_gpus=args.gpus,
    _temp_dir=short_temp_dir
)

# Load the dataset

# Load the specified split from the Hugging Face dataset
hf_dataset = load_dataset("lmsys/lmsys-chat-1m", split=f"train[50:{50 + args.rows}]")

# Convert the Hugging Face dataset to a Ray Dataset and select the 'conversation' column
ds = ray.data.from_huggingface(hf_dataset).select_columns(["conversation"])

print("Dataset loaded.")
print("Rows:", ds.count())

sampling_params = dict(
    temperature=0,
    max_tokens=args.max_tokens,
)

def preprocess(row):
    messages = row["conversation"][:-1]
    input_chars = count_messages_chars(messages)
    
    return {
        "messages": messages,
        "sampling_params": sampling_params,
        "input_chars": input_chars,
    }

def postprocess(row):
    output_chars = count_chars(row.get("generated_text", ""))
    return {
        "answer": row.get("generated_text", ""),
        "input_chars": row.get("input_chars", 0),
        "output_chars": output_chars,
    }

try:
    engine_kwargs = dict(
        enable_chunked_prefill = True,
        max_num_batched_tokens = args.batch_size * 4096,
        max_model_len = 4096,
        max_num_seqs=args.batch_size,
    )

    config = vLLMEngineProcessorConfig(
        model_source=args.model_name,
        engine_kwargs=engine_kwargs,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        trust_remote_code=True,
    )

    processor = build_llm_processor(config, preprocess=preprocess, postprocess=postprocess)

    # Process dataset
    processed_ds = processor(ds)

    metrics.start()

    # This loop executes the processing graph (inference)
    for row in processed_ds.iter_rows():
        metrics.add_row(
            row.get("input_chars", 0),
            row.get("output_chars", 0)
            )
        
    metrics.stop(ds=processed_ds)

    # Print statistics
    metrics.print_stats()
    metrics.save_to_pandas(args)

except Exception as e:
    print(f"Error during processing: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()