"""
DSM Compression Example for LLaMA-2 7B
by Shayan Taherkhani - github : shayanthn --- linkedin : /in/ShayanTaherkhani -- email : shayanthn78@gmail.com
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from dsm_compress import DSMCompressor

# 1. Load Pretrained Model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 2. Initialize DSM Compressor
compressor = DSMCompressor(model, epsilon=1e-6)

# 3. Prepare Calibration Data
def dummy_dataloader(batch_size=4, seq_length=512):
    while True:
        yield {'input_ids': torch.randint(0, 32000, (batch_size, seq_length)),
               'attention_mask': torch.ones((batch_size, seq_length))}

# 4. Compute Scaled Hessians
loss_fn = nn.CrossEntropyLoss()
compressor.compute_scaled_hessian(loss_fn, dummy_dataloader(), iterations=100)

# 5. Generate and Apply Pruning Masks
masks = compressor.generate_pruning_mask()
compressed_model = compressor.apply_compression(masks)

# 6. Save Compressed Model
compressed_model.save_pretrained("llama-7b-dsm-compressed")
tokenizer.save_pretrained("llama-7b-dsm-compressed")

# 7. Verify Performance
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = compressed_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))