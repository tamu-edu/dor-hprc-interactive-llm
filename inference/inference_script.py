from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
LLAMA_8B_MODEL_PATH = "/ztank/scratch/group/hprc/llm_interactive/llm_base_models/llama-3.1-8B-Instruct/"
model_dict = {
        "llama_8B": {
                "model": AutoModelForCausalLM.from_pretrained(LLAMA_8B_MODEL_PATH,device_map="auto"),
                "tokenizer": AutoTokenizer.from_pretrained(LLAMA_8B_MODEL_PATH)
            }
        }
def perform_inference(my_input, max_length, model_name):
    tokenizer = model_dict[model_name]["tokenizer"]
    model = model_dict[model_name]["model"]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    inputs = tokenizer(my_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length, streamer = streamer, pad_token_id=tokenizer.eos_token_id, early_stopping = True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = generated_text[len(my_input):].strip()
    return result

if __name__ == "__main__":
    perform_inference("Hello world in perl", 512, "llama_8B")
