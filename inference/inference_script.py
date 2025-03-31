from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

MODEL_PATH = "/ztank/scratch/group/hprc/llm_interactive/llm_base_models/llama-3.1-8B-Instruct/"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
def perform_inference(my_input):
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    inputs = tokenizer(my_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, streamer = streamer, pad_token_id=tokenizer.eos_token_id, early_stopping = True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input text from the generated output
    result = generated_text[len(my_input):].strip()
    return result

if __name__ == "__main__":
    perform_inference("Hello world in perl")
