# User Documentation
### Cluster Installation
```
git clone --recursive git@github.com:tamu-edu/dor-hprc-interactive-llm.git
cd dor-hprc-interactive-llm
source modules.sh # modify modules.sh if needed to load python 3.10.8
python3 -m venv venv
pip install -r requirements.txt
``` 
you will need to install vllm as well, their installation instructions can be found here:  
https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
### Starting the inference server
First, you will need to define a few environment variables. The following is an example, you will need to change these values for your specific use case:  
```
export NUM_GPUS=4 #The number of gpus to use per inference task, must be less than or equal to the number of gpus on a single node
export IP_LIST_FILE=/sw/hprc/sw/dor-hprc-venv-manager/codeai/child_ips.pkl #The path to the file where the child server ips will be written to. You must have read and write permissions to this file. Users do not need any permissions for this file
export MASTER_IP_ADDRESS_PATH=/sw/hprc/sw/dor-hprc-venv-manager/codeai/ip.pkl #The path where the master server ip address is stored. you must have read and write permissions to this file, users must have read permissions to this file.
export MODEL_PATH=/scratch/group/hprc/llama-models/llama-3_3-70B #path to the model to use, users do not need permissions to this directory.
export NUM_CHILDREN=3 #The number of child instances that will be running (number of app.py instances)
export NUM_TOKENS=1024 #The maximum number of tokens that the model can produce (includes tokens in prompt)
```
Now, with your virtual environment activated, you can run:  
```
cd inference
python3 master_app.py
```
on a node. note that master_app.py does not require a gpu to run.  
You can then run app.py exactly $NUM_CHILDREN times, these processes can be distributed across as many nodes as needed: 
```
python3 app.py //on node a  
python3 app.py //on node b  
...
python3 app.py //on node x
```
Once your app is running, you can test it with:
```
python3 test/test_scale.py
```