from datasets import load_dataset
import os
from Utils.parser import parse_arguments_single_run
from Utils.model import load_model
from Utils.dataset import process_dataset
from Utils.results import save_results, metrics
import importlib
import multiprocessing
import time
import sys

def get_profiler(args, gpus):
    # map your “platform” names to the actual package folders
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
        active_gpus=gpus
    )

def workload(i, args, devices, input_slice, sampling_slice, loaded_flags, ready_flags, done_flags, all_ready, results):
    if args.platform == 'cuda':
       os.environ["CUDA_VISIBLE_DEVICES"] = devices
    else:
        os.environ["HIP_VISIBLE_DEVICES"] = devices
    
    # Load the model
    llm = load_model(args.model_name,
                     args.batch_size,
                     tp=args.tensor_parallel,
                     pp=args.pipeline_parallel,
                     ep=args.expert_parallel,
                     dtype=args.data_type)
    
    loaded_flags[i] = 1
    
    # Warm‐up
    print(f"LLM instance loaded on GPUs {devices}")
    _ = llm.chat(
        input_slice[-50:],
        sampling_params=sampling_slice[-50:],
        use_tqdm=True
    )

    ready_flags[i] = 1

    print(f"GPUs {devices} ready")

    while all_ready.value == 0:
        time.sleep(0.1)

    # Do inference
    _results = llm.chat(
        input_slice,
        sampling_params=sampling_slice,
        use_tqdm=True
    )
    
    results[i] = _results
    done_flags[i] = 1

if __name__ == "__main__":
    args = parse_arguments_single_run()

    total_gpus = int(os.environ['NUM_GPUs'])
    gpus_per_device = total_gpus // args.data_parallel

    assert total_gpus >= args.tensor_parallel * args.pipeline_parallel * args.data_parallel

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:1483]")
    inputs, sampling_params = process_dataset(ds, args.model_name)

    # Splitting dataset according to scaling type

    if args.scaling == "strong":
        samples = len(inputs)
        samples_per_instance = int(samples / args.data_parallel)
        input_slices    = [         inputs[i * samples_per_instance : (i+1) * samples_per_instance] for i in range(args.data_parallel)]
        sampling_slices = [sampling_params[i * samples_per_instance : (i+1) * samples_per_instance] for i in range(args.data_parallel)]

    elif args.scaling == "weak":
        input_slices =    [inputs]          * args.data_parallel
        sampling_slices = [sampling_params] * args.data_parallel

    else:
        print(f"{args.scaling} is not a valid argument for scaling (strong/weak)")
        sys.exit(1)
    
    loaded_flags = multiprocessing.Array('i', args.data_parallel)
    ready_flags  = multiprocessing.Array('i', args.data_parallel)
    done_flags   = multiprocessing.Array('i', args.data_parallel)
    all_ready    = multiprocessing.Value('i', 0)
    all_done     = multiprocessing.Value('i', 0)
    
    for i in range(args.data_parallel):
        loaded_flags[i] = 0
        ready_flags[i] = 0
        done_flags[i] = 0

    manager = multiprocessing.Manager()
    results = manager.list([None] * args.data_parallel)

    processes = []

    print("Starting...")

    # launch instances
    for i in range(args.data_parallel):
        devices = ",".join(str(g) for g in range(gpus_per_device * i,  gpus_per_device * (i+1)))

        process_args = (
            i, args,
            devices,
            input_slices[i],
            sampling_slices[i],
            loaded_flags,
            ready_flags,
            done_flags,
            all_ready,
            results,
            )
        
        process = multiprocessing.Process(target=workload, args=process_args)

        print("starting process")
        process.start()
        processes.append(process)


    # Create and start the profiler
    profiler = get_profiler(args, total_gpus)

    print("Profiler created")

    while all_ready.value == 0:
        if sum([flag for flag in ready_flags]) == args.data_parallel:
            all_ready.value = 1
            print("All GPUs ready. Starting Profiling.")
            profiler.start()
        else:
            time.sleep(0.1)

    while all_done.value == 0:
        if sum([flag for flag in done_flags]) == args.data_parallel:
            all_done.value = 1
            profiler.stop()
            print("All GPUs done.")
        else:
            time.sleep(0.1)

    for process in processes:
        process.join()
    
    results_gathered = []
    for r in results:
        results_gathered += r

    # Profiler data to get latency, ttft, throughput and power metrics
    latency_data, aggregated_data = metrics(results_gathered, profiler)

    # Save the latency results in CSV files
    save_results(args, aggregated_data, './Results/dataset_inference_results.csv')
