import pickle
import requests
def send_question(prompt, index, results):
    response = None
    with open("/sw/hprc/sw/dor-hprc-venv-manager/codeai/ip.pkl", "rb") as my_file:
        ip = pickle.load(my_file)
        url = f"http://{ip}:5000/infer"
        headers = {"Content-Type": "application/json"}
        data = {
        "input": prompt,
        "length": 1024,
        "model": "llama_8B"
        }
        print("sending to url: ", url)
        response = requests.post(url, headers=headers, json=data)
    results[index] = response.json()["response"]

import sys
import threading
from datetime import datetime
def main():
    if(len(sys.argv) < 2):
        print("usage: ", sys.argv[0], " <num child servers>")
        return 1
    num_children = int(sys.argv[1])
    threads = []
    results = [""] * num_children
    start_time = datetime.now()
    for i in range(0, num_children):
        threads.append(threading.Thread(target = send_question, args = ("write a simple pytorch program",i, results), daemon=True))
        threads[i].start()
        print(f"started thread {i}")

    for i in range(0, num_children):
        threads[i].join()
        print(f"thread {i} finished with a total time of {datetime.now() - start_time}")
    
    print("Results: ")
    for value in results:
        print(value)

if __name__ == "__main__":
    main()

