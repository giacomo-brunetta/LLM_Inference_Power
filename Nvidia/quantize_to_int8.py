from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.quantization

# Load model and tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
dtype = torch.float16  # Or change it as needed, e.g., torch.float32
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

# Set model to evaluation mode
model.eval()

# Apply dynamic quantization (int8 on weights)
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Model to quantize
    {torch.nn.Linear},  # Layers to quantize (you can add more layers like LSTM, GRU, etc., if needed)
    dtype=torch.qint8  # Use int8 for weights
)

# Save or use the quantized model for inference
# For example, you can save the quantized model:
torch.save(quantized_model.state_dict(), "quantized_model.pt")

# Optionally, you can use the quantized model for inference as usual
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = quantized_model.generate(inputs['input_ids'], max_length=50)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
