#!/bin/bash
export NUM_CHILDREN=2
export NUM_TOKENS=1024
sbatch ACES_master_job
sbatch ACES_prod_server_job
