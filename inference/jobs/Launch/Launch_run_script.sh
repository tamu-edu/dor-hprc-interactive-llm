#!/bin/bash
export NUM_CHILDREN=2
sbatch Launch_master_job
sbatch Launch_prod_server_job
