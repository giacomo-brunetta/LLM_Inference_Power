from datasets import load_dataset
import os
from Utils.parser import parse_arguments_KL
from Utils.model import load_model
from Utils.dataset import process_dataset
from Utils.results import save_results, metrics
from vllm import SamplingParams
import numpy as np
import copy
import importlib
import multiprocessing
import time
import sys
import contextlib
import gc
import torch
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
import pandas as pd

def report_divergences(model1, model2, data_type_1, data_type_2, kl_2_to_1, kl_1_to_2, js, csv_path = "./Results/KL_divergence.csv"):
     row = {
            "model1": model1,
            "model2": model2,
            "data1": data_type_1,
            "data2": data_type_2,
            "KL(2→1)": kl_2_to_1,
            "KL(1→2)": kl_1_to_2,
            "JS": js,
            }
    
     if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
     else:
        df = pd.DataFrame(columns=row.keys())
    
     if df is None:
        df = pd.DataFrame(columns=row.keys())

     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

     # Print out the values
     print(f"KL({data_type_2} || {data_type_1}) : {kl_2_to_1}")
     print(f"KL({data_type_1} || {data_type_2}) : {kl_1_to_2}")
     print(f"JS({data_type_1} || {data_type_2}) : {js}")

     df.to_csv(csv_path, index=False)
     
def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
         torch.distributed.destroy_process_group()
         gc.collect()
         torch.cuda.empty_cache()

def get_profiler(args, gpus, devices):
    # map your platform names to the actual package folders
    _PLATFORM_MAP = {
        "cuda": "Nvidia",
        "rocm": "AMD",
        "xpu":  "Intel",
        #"hpu":  "Habana",
    }

    try:
        pkg_name = _PLATFORM_MAP[args.platform]
    except KeyError:
        raise ValueError(f"Platform {args.platform!r} not supported")

    # dynamically import the vendor-specific profiler submodule
    module = importlib.import_module(f"{pkg_name}.profiler")

    # grab the Profiler class and instantiate it
    return module.Profiler(
        interval=0.5,
        gpus=gpus,
        active_gpus=gpus,
        watched_devices=devices,
    )

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
    
    v1_fw_pass = llm.chat(io1,
                  sampling_params = one_token,
                  use_tqdm=True)
    
    v2_fw_pass = llm.chat(io2,
                  sampling_params = one_token,
                  use_tqdm=True)
    
    logprobs1 = np.array([get_prompt_cumlogprob(o) for o in v1_fw_pass])
    logprobs2 = np.array([get_prompt_cumlogprob(o) for o in v2_fw_pass])

    return logprobs1, logprobs2

def workload(i, args, devices, input_slice, sampling_slice, gen_flags, done_flags, all_gen, results, logprobs):

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in devices)

    # Load the model
    llm = load_model(args.model_name,
                     args.batch_size,
                     tp=args.tensor_parallel,
                     pp=args.pipeline_parallel,
                     ep=args.expert_parallel,
                     dtype=args.data_type)
    
    # Warm up
    print(f"LLM instance loaded on GPUs {devices}")

    total_gpus = int(os.environ['NUM_GPUs'])

    profiler = get_profiler(args, gpus=total_gpus//2, devices=devices)

    print("Start profiling")
    profiler.start()

    # Do inference
    _results = llm.chat(
        input_slice,
        sampling_params=sampling_slice,
        use_tqdm=True
    )

    profiler.stop()  # Stop the profiler
    try:
        latency_data, aggregated_data = metrics(_results, profiler)
        save_results(args, aggregated_data, './Results/KL_results.csv')
    except Exception as e:
        raise

    results[i] = _results
    gen_flags[i] = 1

    print(f"GPUs {devices} Done")

    while not all_gen.value == 1:
            time.sleep(0.1)

    logprobs0, logprobs1 = get_logprobs(llm, inputs, results[0], results[1])

    logprobs[i].append(logprobs0)
    logprobs[i].append(logprobs1)

    done_flags[i] = 1

if __name__ == "__main__":
    args = parse_arguments_KL()

    total_gpus = int(os.environ['NUM_GPUs'])
    gpus_per_device = total_gpus // 2

    assert total_gpus >= args.tensor_parallel * args.pipeline_parallel * 2

    args1 = copy.deepcopy(args)
    args2 = copy.deepcopy(args)
    args1.model_name = args.model_name_1
    args1.data_type = args.data_type_1
    args2.data_type = args.data_type_2
    args2.model_name = args.model_name_2

    args = [args1, args2]

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:1483]")
    inputs, sampling_params = process_dataset(ds, args[0].model_name)

    # Splitting dataset according to scaling type

    input_slices =    [inputs]           * 2
    sampling_slices = [sampling_params ] * 2
    
    gen_flags  = multiprocessing.Array('i', 2)
    done_flags = multiprocessing.Array('i', 2)
    all_gen    = multiprocessing.Value('i', 0)
    all_done   = multiprocessing.Value('i', 0)

    manager   = multiprocessing.Manager()
    results   = manager.list([ manager.list() for _ in range(2) ])
    logprobs  = manager.list([ manager.list() for _ in range(2) ])
    processes = []


    # launch instances
    for i in range(2):
        devices = list(range(gpus_per_device * i,  gpus_per_device * (i+1)))

        process_args = (
                i, args[i],
                devices,
                input_slices[i],
                sampling_slices[i],
                gen_flags,
                done_flags,
                all_gen,
                results,
                logprobs,
            )
        
        process = multiprocessing.Process(target=workload, args=process_args)

        process.start()
        processes.append(process)

    # Create and start the profiler

    while not all_gen.value == 1:
        if sum([flag for flag in gen_flags]) == 2:
            all_gen.value = 1
            print("All GPUs done with phase 1/2 (generation)")
        else:
            time.sleep(0.1)

    while not all_done.value == 1:
        if sum([flag for flag in done_flags]) == 2:
            all_done.value = 1
            print("All GPUs done with phase 2/2 (KL estimation)")
        else:
            time.sleep(0.1)
    
    # Evaluated with model 1
    gen1_eval1 = logprobs[0][0] # generated with model 1
    gen2_eval1 = logprobs[0][1] # generated with model 2
    # Evaluated with model 2
    gen1_eval2 = logprobs[1][0] # generated with model 1
    gen2_eval2 = logprobs[1][1] # generated with model 2

    kl_1_to_2 = np.mean(gen1_eval1 - gen2_eval1)
    kl_2_to_1 =  np.mean(gen2_eval2 - gen1_eval2)
    
    JSE = (kl_1_to_2 + kl_2_to_1) / 2

    report_divergences(args[0].model_name, args[1].model_name, args[0].data_type, args[1].data_type, kl_2_to_1, kl_1_to_2, JSE)

    for process in processes:
        process.join()
    cleanup()
