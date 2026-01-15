#!/bin/bash
export NUM_CHILDREN=2
export CLUSTER=LAUNCH
export NUM_GPUS=2
export NUM_TOKENS=2048
export IP_LIST_FILE=/sw/hprc/sw/dor-hprc-venv-manager/codeai/child_ips.pkl
export IP_FILE=/sw/hprc/sw/dor-hprc-venv-manager/codeai/ip.pkl
export MASTER_IP_ADDRESS_PATH=$IP_FILE
export MODEL_PATH=/scratch/group/hprc/torch_tune/llm_base_models/llama-3.1-8B-Instruct/
sbatch Launch_master_job
sbatch Launch_prod_server_job
