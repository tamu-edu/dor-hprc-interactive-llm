#!/bin/bash
export VLLM_USE_RAY=true
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export RAY_worker_GPU_ID=-1
python3 inference_script.py
