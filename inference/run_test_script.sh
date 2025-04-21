#!/bin/bash
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export CLUSTER=ACES
python3 inference_script.py
