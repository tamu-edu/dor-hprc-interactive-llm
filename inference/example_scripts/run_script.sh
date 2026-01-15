#!/bin/bash
source ../../modules.sh
source ../../venv/bin/activate
source ./setup.sh
cd ..
python3 ./master_app.py &
python3 ./app.py 1025 #> "logs/${SLURMD_NODENAME}" 2>&1 &
wait
