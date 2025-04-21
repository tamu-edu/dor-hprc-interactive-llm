from vllm import LLM, SamplingParams
import os
cluster = os.environ["CLUSTER"]
MODEL_PATH = None
if(cluster == "ACES"):
    MODEL_PATH = "/scratch/user/u.ks124812/llm_models/llama-8B"
elif(cluster == "LAUNCH"):
    MODEL_PATH = "/ztank/scratch/group/hprc/torch_tune/llm_base_models/llama-3.1-8B-Instruct/"
model_dict = {}
if(cluster == "ACES"):
    model_dict = {
            "llama_8B": LLM(model=MODEL_PATH, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=2, device="xpu")
    }
elif(cluster == "LAUNCH"):
    model_dict = {
            "llama_8B": LLM(model=MODEL_PATH, tensor_parallel_size=2,max_model_len=2048, max_num_seqs=2, device="gpu")
    }

def perform_inference(my_input, max_length, model_name):
    llm = model_dict[model_name]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_length)
    outputs = llm.generate([my_input], sampling_params)
    result = ""
    for output in outputs[0].outputs:
        result += output.text + "\n";
    return result

if __name__ == "__main__":
    result = perform_inference("Hello world in perl", 512, "llama_8B")
    print(result)
