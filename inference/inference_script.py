from transformers import AutoConfig, AutoTokenizer, TextStreamer
import time
import os
from ipex_llm.transformers import AutoModelForCausalLM, init_pipeline_parallel, AutoModel
import torch
init_pipeline_parallel()
NUM_GPUS=int(os.environ["NUM_GPUS"])
LLAMA_8B_MODEL_PATH = "/scratch/user/u.ks124812/llm_models/llama-8B"
# Patch the rope_scaling field to comply with HF expectations
config = AutoConfig.from_pretrained(LLAMA_8B_MODEL_PATH)

"""print("got here")
if hasattr(config, "rope_scaling"):
    print("got here")
    config.rope_scaling = {
        "type": "linear",  # or "dynamic" if needed
        "factor": config.rope_scaling.get("factor", 1.0)
    }
"""
model_dict = {
        "llama_8B": {
                "model":
                AutoModelForCausalLM.from_pretrained(LLAMA_8B_MODEL_PATH,load_in_4bit=True,torch_dtype=torch.float16,optimize_model=True,trust_remote_code=True,use_cache=True,pipeline_parallel_stages=NUM_GPUS),
                "tokenizer": AutoTokenizer.from_pretrained(LLAMA_8B_MODEL_PATH, trust_remote_code=True)
            }
        }
print("made it past model dict")
def perform_inference(prompt, max_length, model_name):
    print("got to inference")
    local_rank = torch.distributed.get_rank()
    model = model_dict[model_name]["model"];
    tokenizer = model_dict[model_name]["tokenizer"]
    with torch.inference_mode():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'xpu:{local_rank}')
        print("got past input ids")
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=max_length)
        print("got past generate")
        # start inference
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=max_length)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        if local_rank == NUM_GPUS -1:
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Inference time: {end-st} s')
            print(f"First token cost {model.first_token_time:.4f} s and rest tokens cost average {model.rest_cost_mean:.4f} s")
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)

if __name__ == "__main__":
    perform_inference("Hello world in perl", 512, "llama_8B")
