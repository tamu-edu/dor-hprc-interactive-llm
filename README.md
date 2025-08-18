# User Documentation
### Cluster Installation
```
git clone --recursive git@github.com:tamu-edu/dor-hprc-interactive-llm.git
cd dor-hprc-interactive-llm
source modules.sh # modify modules.sh if needed to load python 3.10.8
python3 -m venv venv
source venv/bin/activate
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
python3 app.py <port number> //on node a  
python3 app.py <port number> //on node b  
...
python3 app.py <port number> //on node x
```
Note that if you run two app.py instances on the same node, they must have different port numbers.
Once your app is running, you can test it with:
```
python3 test/test_scale.py <num children>
```
### Setting Up Jupyter-AI
Unfortunately several things in the Jupyter-AI extension code are hardcoded: the path to the master ip address and the provider / model names. Luckily, these are not too tricky to change.  
#### Setting the path to the master IP address file
in the file jupyter-ai/packages/HPRC-llama-8B/HPRC_llama_8B/llm.py change the variable master_ip_address_file to the same value you used for the MASTER_IP_ADDRESS_PATH when launching your inference server.  
#### Changing provider name  
in the file jupyter-ai/packages/HPRC-llama-8B/HPRC_llama_8B/provider.py change the name field to a string of your choosing.  
#### Changing model name
in the file jupyter-ai/packages/HPRC-llama-8B/HPRC_llama_8B/llm.py change the model_id field to a string of your choosing. 
