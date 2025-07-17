import argparse
import pandas as pd
import os
from transformers import AutoConfig
import torch
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer 

# Argument parsing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--tp", type=int, default=1, help="Gpus to be used in Tensor Parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Gpus to be used in Pipeline Parallelism")
    parser.add_argument("--ep", action='store_true', help="Expert Parallelism")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model")
    parser.add_argument("--power", action='store_true', help="Measure Power")
    return parser.parse_args()

def parse_arguments_single_run():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--batch_size",       "-bs", type=int,  default=None, help="Batch size")
    parser.add_argument("--data_parallel",    "-dp", type=int,  default=1, help="Gpus to be used in Data Parallelism")
    parser.add_argument("--tensor_parallel",  "-tp", type=int,  default=1, help="Gpus to be used in Tensor Parallelism")
    parser.add_argument("--pipeline_parallel","-pp", type=int,  default=1, help="Gpus to be used in Pipeline Parallelism")
    parser.add_argument("--expert_parallel",  "-ep", type=bool, default=False, help="Enable Expert Parallelism")
    parser.add_argument("--data_type",     "-dtype", type=str,  default='bfloat16')
    parser.add_argument("--model_name",              type=str,  default="meta-llama/Llama-3.1-8B-Instruct", help="Model")
    parser.add_argument("--scaling",           "-s", type=str,  default='strong', help="Options weak, strong")
    parser.add_argument("--platform",          "-p", type=str,  default='cuda', help="Options cuda, rocm, xpu, hpu")
    args = parser.parse_args()
    
    if args.batch_size == -1:
        args.batch_size = None

    return args

def parse_arguments_KL():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--batch_size",       "-bs", type=int,  default=None, help="Batch size")
    parser.add_argument("--tensor_parallel",  "-tp", type=int,  default=1, help="Gpus to be used in Tensor Parallelism")
    parser.add_argument("--pipeline_parallel","-pp", type=int,  default=1, help="Gpus to be used in Pipeline Parallelism")
    parser.add_argument("--expert_parallel",  "-ep", type=bool, default=False, help="Enable Expert Parallelism")
    parser.add_argument("--data_type_1",             type=str,  default='bfloat16')
    parser.add_argument("--data_type_2",             type=str,  default='bfloat16')
    parser.add_argument("--model_name_1",            type=str,  default="meta-llama/Llama-3.1-8B-Instruct", help="Model")
    parser.add_argument("--model_name_2",            type=str,  default="meta-llama/Llama-3.1-8B-Instruct", help="Model")
    parser.add_argument("--platform",          "-p", type=str,  default='cuda', help="Options cuda, rocm, xpu, hpu")
    args = parser.parse_args()
    
    if args.batch_size == -1:
        args.batch_size = None

    return args
