from flask import Flask, request, jsonify
import subprocess
import socket
import os
import pickle
import sys
import time
import json
import requests
app = Flask(__name__)
MASTER_IP_ADDRESS_PATH = os.environ["MASTER_IP_ADDRESS_PATH"]
IP_LIST_FILE = os.environ["IP_LIST_FILE"]
NUM_CHILDREN = int(os.environ["NUM_CHILDREN"])
child_ip_addresses = []
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
    return response

@app.route('/infer', methods=['POST'])
def infer():
    response = None
    try:
        data = request.json
        prompt = data.get("input", "")
        print("prompt: ", prompt)
        model = data.get("model", "")
        max_response_length = int(data.get("length", ""))

        child_ip_addresses = get_children()
        print(child_ip_addresses)
        result = ""
        for ip_address in child_ip_addresses:
            response = send_to_child(ip_address, prompt, max_response_length)
            try:
                status_code = response.json().get("status", None)
            except Exception as e:
                print(f"node {ip_address} down, continuing")
                continue;
            if(status_code and (status_code == 503)):
                print("Child Busy", flush=True)
                continue;
            if(status_code and (status_code == 501)):
                result = "Prompt too long"
            elif(status_code and status_code >= 400):
                continue;
            if(result == ""):
                result = response.json()["response"]
            if(result != ""):
                break;
        if(result == ""):
            result = "All nodes busy, please try again"
        print("response from child was: ", result, flush=True)
        return jsonify({"response": result})
    except Exception as e:
        
        print("got here", flush=True)
        print("failed with exception: ", e, flush=True)
        return jsonify({"status": 500, "error": e, "response": "server busy"})

def get_children():
    file_name = IP_LIST_FILE
    child_ip_addresses = []
    with open(file_name, "rb") as f:
        child_ip_addresses = pickle.load(f)
    return child_ip_addresses

if __name__ == '__main__':
    expected_num_ip_addresses = NUM_CHILDREN
    child_ip_addresses = []
    file_name = IP_LIST_FILE
    with open(file_name, "wb") as f:
        pickle.dump([], f)
    print("waiting for children to write ip addresses to file", flush=True)
    while(len(child_ip_addresses) < expected_num_ip_addresses):
        child_ip_addresses = get_children()
        time.sleep(1)
    result = subprocess.run(
        ["hostname", "-I"],
        capture_output=True,
        text=True,
        check=True
    )
    ip_address = result.stdout.strip().split()[1]
    file_name = MASTER_IP_ADDRESS_PATH
    with open(file_name, "wb") as f:
        pickle.dump(ip_address, f)

    app.run(host='0.0.0.0', port=5000)
