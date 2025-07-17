from datasets import load_dataset
import os
from Utils.parser import parse_arguments_single_run
from Utils.model import load_model
from Utils.dataset import process_dataset
from Utils.results import save_results, metrics
import importlib

def profiler(args, gpus):
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
        gpus=gpus,
        active_gpus=args.tensor_parallel * args.pipeline_parallel
    )


if __name__ == "__main__":
    gpus=int(os.environ['NUM_GPUs'])
    args = parse_arguments_single_run()

    assert gpus >= args.tensor_parallel * args.pipeline_parallel

    model_name = args.model_name
    batch_size = args.batch_size

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:1483]")
    inputs, sampling_params = process_dataset(ds, model_name)

    # Load the model
    llm = load_model(model_name, batch_size,
                     tp=args.tensor_parallel,
                     pp=args.pipeline_parallel,
                     ep=args.expert_parallel,
                     dp=args.data_parallel,
                     dtype=args.data_type)

    # Warm‐up
    print("Warming up...")
    _ = llm.chat(
        inputs[-50:],
        sampling_params=sampling_params[-50:],
        use_tqdm=True
    )

    print("Starting...")

    # Create and start the profiler
    profiler = profiler(args, gpus)
    profiler.start()

    # Do inference
    results = llm.chat(
        inputs,
        sampling_params=sampling_params,
        use_tqdm=True
    )
    profiler.stop()  # Stop the profiler

    # Profiler data to get latency, ttft, throughput and power metrics
    latency_data, aggregated_data = metrics(results, profiler)

    # Save the latency results in CSV files
    latency_data.to_csv(
        f"./Results/Single_Runs/{model_name.split('/')[-1]}_tp{args.tensor_parallel}_{args.pipeline_parallel}_bs{batch_size}{'_ep' if args.expert_parallel else ''}.csv",
        index=True
    )
    save_results(args, aggregated_data, './Results/dataset_inference_results.csv')
