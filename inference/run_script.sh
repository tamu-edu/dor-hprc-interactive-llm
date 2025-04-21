#!/bin/bash
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NUM_GPUS=8
export DEEPSPEED_LOG_LEVEL=debug
which python3
deepspeed --num_gpus $NUM_GPUS inference_script.py --ds_inference
