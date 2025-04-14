from flask import Flask, request, jsonify
import os
import torch
from subprocess import Popen, PIPE
process = Popen(['bash', 'run_script.sh', '-d'], stdout=PIPE, stderr=PIPE,
        stdin=PIPE, text=True)
input_data = "echo Hello from stdin\n"

stdout, stderr = process.communicate(input=input_data, timeout=3600)

print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get("input", "")
        model = data.get("model", "")
        max_response_length = int(data.get("length", ""))
        return jsonify({"response": perform_inference(prompt, max_response_length, model)})
    except Exception as e:
        print("failed with exception: ", e)
        return jsonify({"status": 500})

if __name__ == '__main__':
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        app.run(host='0.0.0.0', port=5000)
    else:
        torch.distributed.barrier()
