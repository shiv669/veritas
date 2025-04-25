import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Load model and tokenizer
model_path = os.path.abspath(".")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
model.eval()

# Inference function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs="text",
    title="🔮 Mistral 7B Chat",
    description="Running locally on your MacBook M4"
).launch(True)
