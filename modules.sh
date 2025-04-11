#!/bin/bash
ml GCC/13.2.0
ml Python/3.11.5
source /sw/hprc/sw/oneAPI/2024.0/setvars.sh
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9090
export FI_PROVIDER=tcp
export USE_XETLA=OFF
export OMP_NUM_THREADS=6
export IPEX_LLM_QUANTIZE_KV_CACHE=1

if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0
