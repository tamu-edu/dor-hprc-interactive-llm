import os
import time
import torch
import deepspeed
from transformers import AutoConfig, AutoTokenizer, pipeline, AutoModelForCausalLM

NUM_GPUS = int(os.environ["NUM_GPUS"])
local_rank = int(os.getenv("LOCAL_RANK", "0"))
print("local rank is: ", local_rank)
LLAMA_8B_MODEL_PATH = "/scratch/user/u.ks124812/llm_models/llama-8B"

config = AutoConfig.from_pretrained(LLAMA_8B_MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(LLAMA_8B_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(LLAMA_8B_MODEL_PATH)
ds_engine = deepspeed.init_inference(
    model,
    mp_size=NUM_GPUS,
    dtype=torch.float16,
    replace_with_kernel_inject=False,
)

model = ds_engine.module

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=local_rank)

def perform_inference(prompt, max_length, model):
    print("Performing inference for prompt:", prompt)
    output = pipe(prompt, max_new_tokens=max_length)
    return output

if __name__ == "__main__":
    prompt = "What is tensor parallelism and why is it important in large language models?"
    output = perform_inference(prompt, max_length=100)
    print("Output:", output)

