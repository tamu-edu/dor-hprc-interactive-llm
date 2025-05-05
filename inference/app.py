from flask import Flask, request, jsonify
import subprocess
from inference_script import perform_inference
import socket
import pickle
app = Flask(__name__)
@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get("input", "")
        print("prompt: ", prompt)
        model = data.get("model", "")
        max_response_length = int(data.get("length", ""))
        if(max_response_length > 512):
            return jsonify({"status": 500, "error": "max length too long"})
        return jsonify({"response": perform_inference(prompt, max_response_length, model)})
    except Exception as e:
        print("got here")
        print("failed with exception: ", e)
        return jsonify({"status": 500, "error": e})
if __name__ == '__main__':
    result = subprocess.run(
        ["hostname", "-I"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout.strip().split())
    ip_address = result.stdout.strip().split()[0]
    file_name = "/sw/hprc/sw/dor-hprc-venv-manager/codeai/ip.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(ip_address, f)
    app.run(host='0.0.0.0', port=5000)
