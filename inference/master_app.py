from flask import Flask, request, jsonify
import subprocess
import socket
import pickle
import sys
import time
import requests
app = Flask(__name__)
TIMEOUT = 3
child_ip_addresses = []
def check_if_child_responsive(ip_address):
    url = f"http://{ip_address}/alive"
    try:
        print("sending request to ", url, flush=True)
        response = requests.get(url, timeout=TIMEOUT)
        print("response status: ", response.status_code)
        if(response):
            return True
        return False
    except requests.RequestException as e:
        print(f"Request to {ip_address} failed: {e}", flush=True)
        return False

def send_to_child(ip_address, prompt, length):
    url = f"http://{ip_address}/infer"
    headers = {"Content-Type": "application/json"}
    data = {
        "input": prompt,
        "length": length,
        "model": "llama_8B"
    }
    print("sending to url: ", url)
    response = requests.post(url, headers=headers, json=data)
    return response.json()["response"]

@app.route('/infer', methods=['POST'])
def infer():
    try:
        data = request.json
        prompt = data.get("input", "")
        print("prompt: ", prompt)
        model = data.get("model", "")
        max_response_length = int(data.get("length", ""))
        if(max_response_length > 512):
            return jsonify({"status": 400, "error": "max length too long"})

        result = ""
        for ip_address in child_ip_addresses:
            alive = check_if_child_responsive(ip_address)
            if(alive):
                print("address was alive", flush=True)
                result = send_to_child(ip_address, prompt, max_response_length)
                break;
        if(result == ""):
            result = "All nodes busy, please try again"
        return jsonify({"response": result})
    except Exception as e:
        print("got here")
        print("failed with exception: ", e)
        return jsonify({"status": 500, "error": e, "response": "server busy"})

if __name__ == '__main__':
    expected_num_ip_addresses = int(sys.argv[1]) 
    child_ip_addresses = []
    file_name = "/sw/hprc/sw/dor-hprc-venv-manager/codeai/child_ips.pkl"
    with open(file_name, "wb") as f:
        pickle.dump([], f)
    print("waiting for children to write ip addresses to file", flush=True)
    while(len(child_ip_addresses) < expected_num_ip_addresses):
        with open(file_name, "rb") as f:
            child_ip_addresses = pickle.load(f)
        time.sleep(1)
    print("child ip addresses: ", child_ip_addresses, flush=True)  
    result = subprocess.run(
        ["hostname", "-I"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout.strip().split())
    ip_address = result.stdout.strip().split()[1]
    print(ip_address)
    file_name = "/sw/hprc/sw/dor-hprc-venv-manager/codeai/ip.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(ip_address, f)
    app.run(host='0.0.0.0', port=5000)
