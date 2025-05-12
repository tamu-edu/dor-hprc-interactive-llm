#!/bin/bash
export NUM_CHILDREN=2
sbatch ACES_master_job
sbatch ACES_prod_server_job
