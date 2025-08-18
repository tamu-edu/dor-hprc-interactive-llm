#!/bin/bash
export NUM_GPUS=2 #The number of gpus to use per inference task, must be less than or equal to the number of gpus on a single node
export IP_LIST_FILE="${HOME}/child_ips.pkl" #The path to the file where the child server ips will be written to. You must have read and write permissions to this file. Users do not need any permissions for this file
export MASTER_IP_ADDRESS_PATH="${HOME}/ip.pkl" #The path where the master server ip address is stored. you must have read and write permissions to this file, users must have read permissions to this file.
export MODEL_PATH=/ztank/scratch/group/hprc/torch_tune/llm_base_models/llama-3.1-8B-Instruct/ #path to the model to use, users do not need permissions to this directory.
export NUM_CHILDREN=1 #The number of child instances that will be running (number of app.py instances)
export NUM_TOKENS=1024 #The maximum number of tokens that the model can produce (includes tokens in prompt)

