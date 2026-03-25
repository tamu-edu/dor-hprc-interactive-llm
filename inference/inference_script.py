import os
import random
import subprocess
import multiprocessing as mp
import re
import torch
from vllm import LLM, SamplingParams


def remove_ansi_sequences(text):
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def extract_final_answer(text):
    """
    Parse model output of the form:

        analysis...
        assistantfinal<final answer>

    or variants with whitespace/newlines.

    If that pattern is not found, return the raw text.
    """
    if not text:
        return text

    cleaned = remove_ansi_sequences(text).strip()

    # Most direct case: everything after 'assistantfinal'
    match = re.search(r"assistantfinal\s*(.*)$", cleaned, flags=re.DOTALL)
    if match:
        final = match.group(1).strip()
        if final:
            return final

    # Slightly more permissive fallback in case there is whitespace between words
    match = re.search(r"assistant\s*final\s*(.*)$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        final = match.group(1).strip()
        if final:
            return final

    return cleaned


def validate_length(llm, text, max_tokens):
    tokenizer = llm.get_tokenizer()
    input_tokens = tokenizer.encode(text)
    return len(input_tokens) <= max_tokens


def build_models():
    cluster = os.environ.get("CLUSTER", None)
    num_gpus = int(os.environ["NUM_GPUS"])
    model_path = os.environ["MODEL_PATH"].rstrip("/")
    max_output_tokens = int(os.environ["NUM_OUTPUT_TOKENS"])
    max_input_tokens = int(os.environ["MAX_INPUT_TOKENS"])

    print("MAX_OUTPUT_TOKENS:", max_output_tokens, flush=True)
    print("MAX_INPUT_TOKENS:", max_input_tokens, flush=True)

    if cluster == "ACES":
        return {
            "llama_8B": LLM(
                model=model_path,
                dtype="bfloat16",
                enforce_eager=True,
                tensor_parallel_size=num_gpus,
                max_model_len=max_output_tokens + max_input_tokens,
                max_num_seqs=1,
                trust_remote_code=True,
            )
        }
    else:
        return {
            "llama_8B": LLM(
                model=model_path,
                tensor_parallel_size=num_gpus,
                max_model_len=max_output_tokens + max_input_tokens,
                max_num_seqs=2,
            )
        }


def perform_inference(model_dict, user_text, max_length, model_name):
    llm = model_dict[model_name]
    tokenizer = llm.get_tokenizer()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. "
                "For simple factual questions, answer directly in one short sentence. "
                "Do not explain your reasoning unless explicitly asked. "
            ),
        },
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        max_tokens=max_length,
        min_tokens=1,
    )

    outputs = llm.generate([prompt], sampling_params)
    raw_text = outputs[0].outputs[0].text
    return extract_final_answer(raw_text)


def main():
    vllm_host_ip = subprocess.check_output(
        "hostname -I | awk '{print $1}'", shell=True
    ).decode().strip()
    os.environ["VLLM_HOST_IP"] = vllm_host_ip

    port = random.randint(1024, 65535)
    os.environ["VLLM_PORT"] = str(port)
    print("vllm using port", port, flush=True)

    print("has torch.xpu:", hasattr(torch, "xpu"), flush=True)
    print(
        "xpu available:",
        (torch.xpu.is_available() if hasattr(torch, "xpu") else None),
        flush=True,
    )

    model_dict = build_models()
    result = perform_inference(model_dict, "write hello world in perl", 512, "llama_8B")
    print(result, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
