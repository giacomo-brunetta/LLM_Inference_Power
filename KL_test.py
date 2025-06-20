from datasets import load_dataset
import os
from Utils.parser import parse_arguments_single_run
from Utils.model import load_model
from Utils.dataset import process_dataset

if __name__ == "__main__":
    gpus=int(os.environ['NUM_GPUs'])
    args = parse_arguments_single_run()

    assert gpus >= args.tensor_parallel * args.pipeline_parallel

    model_name = args.model_name
    batch_size = args.batch_size

    # Load the dataset (we pick 1,483 so that after filtering we end up with 1,000)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train[0:10]")
    inputs, sampling_params = process_dataset(ds, model_name)

    # Load the model
    llm = load_model(model_name, batch_size,
                     tp=args.tensor_parallel,
                     pp=args.pipeline_parallel,
                     ep=args.expert_parallel,
                     dp=args.data_parallel,
                     dtype=args.data_type)

    results = llm.chat(
        inputs[:10],
        sampling_params=sampling_params[:10],
        use_tqdm=True
    )

    print(results)
