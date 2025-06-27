from datasets import Dataset
from transformers import AutoTokenizer 
from vllm import SamplingParams

def chat_len(chat):
    return sum([len(msg['content']) for msg in chat])

def process_dataset(ds, model_name, logprobs = False):
    """
    Take a chat dataset as input and return input messages ans Sampling Parameters for the result
    """
    inputs = []
    outputs = []

    for chat in ds["conversation"]:
        # Take everything except last message as "input", last message as "response"
        input_chat = chat[:-1]
        response = chat[-1]

        in_len = chat_len(input_chat)
        out_len = chat_len([response])

        # Only keep examples whose input/output lengths are between 64 and 20,000 chars
        if 64 <= in_len <= 20000 and 64 <= out_len <= 20000:
            inputs.append(input_chat)
            outputs.append(response)

    # Create a small Dataset from the filtered outputs
    outputs_ds = Dataset.from_list(outputs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def len_in_tokens(batch):
        enc = tokenizer(
            batch["content"],        # this is a list of strings
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        # enc["input_ids"] is now a list-of-lists, one per example in the batch.
        # We need to return a list of lengths, not just a single int.
        return {"out_len": [len(ids) for ids in enc["input_ids"]]}

    out_lengths_ds = outputs_ds.map(
        len_in_tokens,
        batched=True,
        batch_size=256,
        num_proc=64
    )

    out_lengths = out_lengths_ds["out_len"]
    sampling_params = [
        SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=ol,
        )
        for ol in out_lengths
    ]

    # Make sure we still have exactly 1,000 examples after filtering
    # assert len(inputs) == len(sampling_params) and len(inputs) == 1000

    return inputs, sampling_params
