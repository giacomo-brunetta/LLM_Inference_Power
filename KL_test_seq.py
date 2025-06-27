from datasets import load_dataset
import os
from Utils.parser import parse_arguments_single_run
from Utils.model import load_model
from Utils.dataset import process_dataset
from Utils.results import get_latency_data
import importlib
import multiprocessing
import time
import sys
from vllm import SamplingParams
import numpy as np
import copy

def concat(input, answers):
    return [i + [{"content": o.outputs[0].text, "role": "assistant"}] for i,o in zip(input,answers)]

def get_prompt_cumlogprob(request, start=1):
    prompt_logprobs = [
        lp[token].logprob
        for lp,token in zip(request.prompt_logprobs[start:],request.prompt_token_ids[start:])
        ]

    return sum(prompt_logprobs)

def get_logprobs(llm, inputs, output1, output2):

    # Generate one token to get prompt logprobs
    one_token = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=1,
            prompt_logprobs=1,
            logprobs=1,
    )
    
    # concat messages according to OpenAI chat format
    io1 = concat(inputs, output1)
    io2 = concat(inputs, output2)

    v1_fw_pass = llm.chat( io1,
                  sampling_params = one_token,
                  use_tqdm=True)
    
    v2_fw_pass = llm.chat( io2,
                  sampling_params = one_token,
                  use_tqdm=True)
    
    logprobs1 = np.array([get_prompt_cumlogprob(o) for o in v1_fw_pass])
    logprobs2 = np.array([get_prompt_cumlogprob(o) for o in v2_fw_pass])

    return logprobs1, logprobs2

def workload(i, args, input_slice, sampling_slice, results, logprobs):

    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    # Load the model
    llm = load_model(args.model_name,
                     args.batch_size,
                     tp=args.tensor_parallel,
                     pp=args.pipeline_parallel,
                     ep=args.expert_parallel,
                     dtype=args.data_type)
    
    # Warm‐up
    print(f"LLM instance loaded on GPUs {devices}")

    # Do inference
    _results = llm.chat(
        input_slice,
        sampling_params=sampling_slice,
        use_tqdm=True
    )

    _,_ = get_latency_data(_results)
    
    results[i] = _results
    gen_flags[i] = 1

    print(f"GPUs {devices} Done")

    # The process has the instance of the original model
    while not all_gen.value == 1:
            time.sleep(0.1)

    i_other = 1 if i == 0 else 0
    other_results = results[i_other]
    
    logprobs0, logprobs1 = get_logprobs(llm, inputs, _results, other_results)

    logprobs[i].append(logprobs0)
    logprobs[i].append(logprobs1)

    done_flags[i] = 1

if __name__ == "__main__":
    args = parse_arguments_single_run()

    total_gpus = int(os.environ['NUM_GPUs'])
    gpus_per_device = total_gpus // 2

    assert total_gpus >= args.tensor_parallel * args.pipeline_parallel * 2

    args1 = copy.deepcopy(args)
    args2 = copy.deepcopy(args)
    #args2.model_name = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    #args2.model_name = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
    args1.model_name = "Qwen/Qwen3-8B"
    args2.model_name = "Qwen/Qwen3-8B"
    #args2.data_type  = "compressed"

    args = [args1, args2]

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:1483]")
    inputs, sampling_params = process_dataset(ds, args[0].model_name)

    # Splitting dataset according to scaling type


    for i in range(2):
        # Load the model
        llm = load_model(args[i].model_name,
                     args[i].batch_size,
                     tp=args[i].tensor_parallel,
                     pp=args[i].pipeline_parallel,
                     ep=args[i].expert_parallel,
                     dtype=args[i].data_type)


        # Do inference
        _results = llm.chat(
                inputs,
                sampling_params=sampling_params,
                use_tqdm=True
            )

        _,_ = get_latency_data(_results)

        results[i] = _results
    
    for i in range(2):
        # Load the model
        llm = load_model(args[i].model_name,
                     args[i].batch_size,
                     tp=args[i].tensor_parallel,
                     pp=args[i].pipeline_parallel,
                     ep=args[i].expert_parallel,
                     dtype=args[i].data_type)
        
        i_other = 1 if i == 0 else 0

        logprobs0, logprobs1 = get_logprobs(llm, inputs, results[i], results[i_other])

        logprobs[i].append(logprobs0)
        logprobs[i].append(logprobs1)


    # The process has the instance of the original model
    while not all_gen.value == 1:
            time.sleep(0.1)

    i_other = 1 if i == 0 else 0
    other_results = results[i_other]

    logprobs0, logprobs1 = get_logprobs(llm, inputs, _results, other_results)

    logprobs[i].append(logprobs0)
    logprobs[i].append(logprobs1)



    E0_p0 = logprobs[0][0]
    E0_p1 = logprobs[0][1]
    E1_p1 = logprobs[1][0]
    E1_p0 = logprobs[1][1]

    KL_0 = np.mean(E0_p0 - E0_p1)
    KL_1 = np.mean(E1_p0 - E1_p1)
    JSE = 0.5 * (KL_0 + KL_1)

    print("="*50)
    print("KL quant || base ) : ",KL_0)
    print("KL base  || quant) : ",KL_1)
    print("JS(quant || base ) : ",JSE)
    print("="*50)
