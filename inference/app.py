import pickle
import subprocess
import sys
from flask import Flask, request, jsonify
import portalocker
import os
from inference_script import perform_inference

app = Flask(__name__)

port = sys.argv[1]
result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
addresses = result.stdout.strip().split()
ip = addresses[1]
instance_id = ip.replace('.', '_') + "_" + port
LOCK_FILE = f"/tmp/infer_lock_{instance_id}.lock"
IP_LIST_FILE = os.environ["IP_LIST_FILE"]
MAX_TOKENS = int(os.environ["NUM_TOKENS"])

def append_ip_to_file(file_path, ip_address):
    try:
        open(file_path, 'rb').close()
    except FileNotFoundError:
        with open(file_path, 'wb') as f:
            pickle.dump([], f)
    with open(file_path, "r+b") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        ip_list = pickle.load(f)
        if ip_address not in ip_list:
            ip_list.append(ip_address)
            f.seek(0)
            pickle.dump(ip_list, f)
            f.truncate()
        portalocker.unlock(f)

append_ip_to_file(IP_LIST_FILE, f"{ip}:{port}")

def try_acquire_lock():
    f = open(LOCK_FILE, "w")
    try:
        portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
        return f
    except portalocker.exceptions.LockException:
        f.close()
        return None

def release_lock(f):
    portalocker.unlock(f)
    f.close()

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get("input", "")
        model = data.get("model", "")
        max_len = int(data.get("length", "0"))
        if max_len > MAX_TOKENS:
            return jsonify({"status": 501, "error": "max length too long"})
        lock_f = try_acquire_lock()
        if not lock_f:
            return jsonify({"status": 503, "error": "Server is busy, try again later", "response": "Server is busy"})
        try:
            try:
                resp = perform_inference(prompt, max_len, model)
            except ValueError:
                return jsonify({"status": 501, "response": "max length too long"})
            return jsonify({"response": resp})
        finally:
            release_lock(lock_f)
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port))

