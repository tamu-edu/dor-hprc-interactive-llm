from vllm import LLM, SamplingParams
LLAMA_8B_MODEL_PATH = "/ztank/scratch/group/hprc/torch_tune/llm_base_models/llama-3.1-8B-Instruct/"

model_dict = {
        "llama_8B": LLM(model=LLAMA_8B_MODEL_PATH, tensor_parallel_size=2, max_model_len=2048, max_num_seqs=2)
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
