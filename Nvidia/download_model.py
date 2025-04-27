import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    print(f"Loading model '{model_name}' from Hugging Face...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    end_time = time.time()
    print(f"Model and tokenizer loaded in {end_time - start_time:.2f} seconds.")

    return model, tokenizer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    model, tokenizer = load_model(model_name)