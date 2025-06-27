import os
import torch
from vllm import LLM
from transformers import AutoConfig

# Constent model loading between runs

def cap_max_model_len(model_name, value):
    config = AutoConfig.from_pretrained(model_name)
    supported_max_len = (
        config.max_position_embeddings
        or config.model_max_length
        or 0
    )
    return min(supported_max_len, value)

def get_vocab_size(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return config.vocab_size

def load_model(model_name, batch_size, dtype='bfloat16', tp = 1, pp = 1, ep = False, dp = 1, use_v1=False):
    """
    Load the model using vLLM. The GPUs are set to be used in tensor parallelism.
    """
    max_model_len = cap_max_model_len(model_name, 10000)
    
    print("="*50)
    print(f"Loading {model_name}")
    print(f"Data Type         : {dtype}")
    print(f"Tensor Parallel   : {tp}")
    print(f"Pipeline Parallel : {pp}")
    print(f"Expert Parallel   : {ep}")
    print(f"Data Parallel     : {dp}")
    print(f"Batch Size        : {batch_size}")
    print(f"Max sequence len  : {max_model_len}")
    print("="*50)
    print()

    torch._dynamo.config.suppress_errors = True # Suppress errors
    os.environ["VLLM_USE_V1"] = str(1 if use_v1 else 0)  # Per-request profiling is not in v1 engine, so use v0
    
    if dtype == 'fp8':
        llm = LLM(
            model=model_name,
            max_num_seqs = batch_size,
            tokenizer=None,
            quantization="fp8",
            kv_cache_dtype="auto",
            tensor_parallel_size = tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel= ep,
            trust_remote_code=True,
            enforce_eager=True,
            device='cuda',
            block_size=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.9,
            load_format='auto',
            distributed_executor_backend=None,
            enable_prefix_caching=False,
            disable_sliding_window=False,
            max_model_len=max_model_len,
        )
    
    elif dtype == 'compressed':
        llm = LLM(
            model=model_name,
            max_num_seqs=batch_size,
            tokenizer=None,
            quantization="compressed-tensors",
            kv_cache_dtype="auto",
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
            trust_remote_code=True,
            enforce_eager=True,
            device='cuda',
            block_size=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.9,
            load_format='auto',
            distributed_executor_backend=None,
            enable_prefix_caching=False,
            disable_sliding_window=False,
            max_model_len=max_model_len,
        )
    
    elif dtype == 'gptq':
        llm = LLM(
            model=model_name,
            max_num_seqs=batch_size,
            tokenizer=None,
            quantization="gptq",
            kv_cache_dtype="auto",
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
            trust_remote_code=True,
            enforce_eager=True,
            device='cuda',
            block_size=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.9,
            load_format='auto',
            distributed_executor_backend=None,
            enable_prefix_caching=False,
            disable_sliding_window=False,
            max_model_len=max_model_len,
        )

    else:
        llm = LLM(
            model=model_name,
            max_num_seqs = batch_size,
            tokenizer=None,
            quantization=None,
            tensor_parallel_size = tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel= ep,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=True,
            kv_cache_dtype='auto',
            device='cuda',
            block_size=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization= 0.9,
            load_format='auto',
            distributed_executor_backend=None,
            enable_prefix_caching=False,
            disable_sliding_window=False,
            max_model_len=max_model_len,
        )

    return llm
