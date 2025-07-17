from datasets import load_dataset
import os
from Utils.parser import parse_arguments_single_run
from Utils.model import load_model
from Utils.dataset import process_dataset
from Utils.results import save_results, metrics
import importlib
import time
import sys
import ray

def get_profiler(args, gpus):
    # map your “platform” names to the actual package folders
    _PLATFORM_MAP = {
        "cuda": "Nvidia",
        "rocm": "AMD",
        "xpu":  "Intel",
        #"hpu": "Habana",
        #"tpu": "Google",
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

@ray.remote(num_cpus=8, num_gpus=1)
def workload(i, args, devices, input_slice, sampling_slice,
             loaded_flags, ready_flags, done_flags, all_ready):
    
    assigned = ray.get_gpu_ids()
    print(f"Worker {i} assigned GPUs:", assigned)

    # Load model once per task
    llm = load_model(
        args.model_name,
        args.batch_size,
        tp=args.tensor_parallel,
        pp=args.pipeline_parallel,
        ep=args.expert_parallel,
        dtype=args.data_type,
        xpu=(args.platform == "xpu")
    )

    # Signal loaded
    ray.get(loaded_flags[i].set.remote())

    # Warm-up and signal ready
    _ = llm.chat(input_slice[-50:], sampling_params=sampling_slice[-50:], use_tqdm=True)
    ray.get(ready_flags[i].set.remote())
    print(f"GPUs {devices} ready")

    # Wait for driver to set all_ready
    while not ray.get(all_ready.get.remote()):
        time.sleep(0.1)

    # Inference
    results = llm.chat(input_slice, sampling_params=sampling_slice, use_tqdm=True)

    # Signal done
    ray.get(done_flags[i].set.remote())
    return results

@ray.remote
class SingleFlag:
    def __init__(self):
        self.flag = 0
    def set(self, v=1):
        self.flag = v
    def get(self):
        return self.flag

if __name__ == "__main__":
    args = parse_arguments_single_run()

    total_gpus = int(os.environ['NUM_GPUs'])
    gpus_per_device = total_gpus // args.data_parallel

    assert total_gpus >= args.tensor_parallel * args.pipeline_parallel * args.data_parallel

    if args.platform == 'xpu':
        ray.init(address="auto")
        print("Available before scheduling:", ray.available_resources())

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
    
    loaded_flags = [SingleFlag.remote() for _ in range(args.data_parallel)]
    ready_flags  = [SingleFlag.remote() for _ in range(args.data_parallel)]
    done_flags   = [SingleFlag.remote() for _ in range(args.data_parallel)]
    all_ready    = SingleFlag.remote()
    all_done     = SingleFlag.remote()

    print("Starting...")

    futures = []
    for i in range(args.data_parallel):
        # choose a contiguous GPU slice for this worker
        gpus_per_dev = args.tensor_parallel
        devices = ",".join(str(g) for g in range(
            gpus_per_dev * i, gpus_per_dev * (i + 1)
        ))

        fut = workload.remote(
            i, args, devices,
            input_slices[i], sampling_slices[i],
            loaded_flags, ready_flags, done_flags,
            all_ready
        )
        futures.append(fut)

    # wait until every worker has set its loaded flag
    while True:
        vals = ray.get([lf.get.remote() for lf in loaded_flags])
        if all(vals):
            break
        time.sleep(0.1)
    print("All workers loaded → signalling ready")

    ray.get(all_ready.set.remote(1))

    # wait until every worker has set its done flag
    while True:
        vals = ray.get([df.get.remote() for df in done_flags])
        if all(vals):
            break
        time.sleep(0.1)
    print("All workers done")
    
    ray.get(all_done.set.remote(1))

    # gather outputs
    results = ray.get(futures)

    if args.platform == "xpu":
        ray.shutdown()