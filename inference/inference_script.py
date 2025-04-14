import os
import time
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
import sys
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

def perform_inference(prompt, max_length, model = None):
    print("Performing inference for prompt:", prompt)
    output = pipe(prompt, max_new_tokens=max_length)
    return output

def broadcast_prompt(prompt):
    """
    Broadcasts an object (here, a prompt string) from rank 0 to all ranks.
    We use broadcast_object_list for simplicity.
    """
    # On rank 0 the prompt_list contains the real prompt;
    # on other ranks, it’s an empty string placeholder.
    prompt_list = [prompt] if local_rank == 0 else [""]
    dist.broadcast_object_list(prompt_list, src=0)
    return prompt_list[0]

if __name__ == "__main__":
    
    if local_rank == 0:
        print(f"[rank {local_rank}] Waiting for multi-line input (end your prompt with <<<END>>>):", flush=True)
        buffer = []
        for line in sys.stdin:
            if "<<<END>>>" in line:
                clean_line = line.replace("<<<END>>>", "").strip()
                buffer.append(clean_line)
                prompt = "\n".join(buffer).strip()
                buffer = []  
                if prompt.lower() == "exit":
                    prompt = "exit"
                    broadcast_prompt(prompt)
                    print(f"[rank {local_rank}] Exiting.", flush=True)
                    break
                prompt = broadcast_prompt(prompt)
                try:
                    response = perform_inference(prompt, max_length=100)
                    print(response, flush=True)
                except Exception as e:
                    print(f"[rank {local_rank}] Inference failed: {e}", flush=True)
            else:
                buffer.append(line.rstrip("\n"))
    else:
        while True:
            prompt = broadcast_prompt("")  
            if prompt.lower() == "exit":
                break
            try:
                _ = perform_inference(prompt, max_length=100)
            except Exception as e:
                print(f"[rank {local_rank}] Error during inference: {e}", flush=True)
