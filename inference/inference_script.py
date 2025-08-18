import os
import random
import subprocess
VLLM_HOST_IP = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True).decode().strip()
os.environ["VLLM_HOST_IP"] = VLLM_HOST_IP
port = random.randint(1024, 65535)
print("vllm using port ", port, flush=True)
os.environ["VLLM_PORT"] = str(port)
from vllm import LLM, SamplingParams
import re
cluster = os.environ.get("CLUSTER", None)
NUM_GPUS = int(os.environ["NUM_GPUS"])
MODEL_PATH = os.environ["MODEL_PATH"]
MAX_TOKENS = int(os.environ["NUM_TOKENS"])

model_dict = {}
if(cluster == "ACES"): #because ACES uses xpus
    model_dict = {
            "llama_8B":LLM(model=MODEL_PATH,dtype="bfloat16",enforce_eager=True,tensor_parallel_size=NUM_GPUS,max_model_len=MAX_TOKENS, max_num_seqs=1, device="xpu")
    }
else:
    model_dict = {
            "llama_8B": LLM(model=MODEL_PATH, tensor_parallel_size=NUM_GPUS,max_model_len=MAX_TOKENS, max_num_seqs=2)
    }
def remove_ansi_sequences(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def validate_length(llm, text, max_tokens):
    tokenizer = llm.get_tokenizer()
    input_tokens = tokenizer.encode(text)
    num_tokens = len(input_tokens)
    if(num_tokens > max_tokens):
        return False
    return True

def perform_inference(my_input, max_length, model_name):
    my_input = remove_ansi_sequences(my_input)
    llm = model_dict[model_name]
    short_enough = validate_length(llm, my_input, max_length)
    if(not short_enough):
        raise ValueError("Prompt too long")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=max_length, min_tokens = 5)
    print("my input: ", my_input)
    outputs = llm.generate([my_input], sampling_params)
    result = ""
    for output in outputs[0].outputs:
        result += output.text + "\n";
    print("returning result: ", result, flush=True)
    return result

if __name__ == "__main__":
    result = perform_inference("write hello world in perl", 512, "llama_8B")
    print(result)
