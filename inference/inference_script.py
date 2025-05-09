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
cluster = os.environ["CLUSTER"]
MODEL_PATH = None
if(cluster == "ACES"):
    MODEL_PATH = "/scratch/group/hprc/llama-models/llama-3_3-70B"
elif(cluster == "LAUNCH"):
    MODEL_PATH = "/ztank/scratch/group/hprc/torch_tune/llm_base_models/llama-3.1-8B-Instruct/"
model_dict = {}
if(cluster == "ACES"):
    NUM_GPUS = int(os.environ["NUM_GPUS"])
    model_dict = {
            "llama_8B":LLM(model=MODEL_PATH,dtype="bfloat16",enforce_eager=True,tensor_parallel_size=NUM_GPUS,max_model_len=1024, max_num_seqs=1, device="xpu")
    }
elif(cluster == "LAUNCH"):
    model_dict = {
            "llama_8B": LLM(model=MODEL_PATH, tensor_parallel_size=2,max_model_len=2048, max_num_seqs=2, device="cuda")
    }
def remove_ansi_sequences(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def perform_inference(my_input, max_length, model_name):
    my_input = remove_ansi_sequences(my_input)
    llm = model_dict[model_name]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=max_length, min_tokens = 5)
    print("my input: ", my_input)
    print("max length: ", max_length)
    outputs = llm.generate([my_input], sampling_params)
    print(outputs, flush=True)
    result = ""
    for output in outputs[0].outputs:
        print(output)
        result += output.text + "\n";
    print("returning result: ", result, flush=True)
    return result

if __name__ == "__main__":
    result = perform_inference("write hello world in perl", 512, "llama_8B")
    print(result)
