#!/bin/bash
export NUM_GPUS=8
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS inference_script.py 
