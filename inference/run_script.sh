#!/bin/bash
export NUM_GPUS=8
export DEEPSPEED_LOG_LEVEL=debug
deepspeed --num_gpus $NUM_GPUS inference_script.py --ds_inference

