#!/bin/bash
export NUM_GPUS=8
deepspeed --num_gpus $NUM_GPUS inference_script.py --ds_inference

