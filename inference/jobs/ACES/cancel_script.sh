#!/bin/bash

# Script to cancel all jobs for user u.ks124812

USER="u.ks124812"

echo "Checking for jobs for user: $USER"

# Get job IDs from squeue output, skipping the header line
JOB_IDS=$(squeue -u "$USER" | awk 'NR>1 {print $1}')

# Check if any jobs were found
if [ -z "$JOB_IDS" ]; then
    echo "No jobs found for user $USER"
    exit 0
fi

echo "Found the following job IDs:"
echo "$JOB_IDS"

# Cancel each job
for JOB_ID in $JOB_IDS; do
    echo "Canceling job $JOB_ID..."
    scancel "$JOB_ID"

    # Check if scancel was successful
    if [ $? -eq 0 ]; then
        echo "Successfully canceled job $JOB_ID"
    else
        echo "Failed to cancel job $JOB_ID"
    fi
done

echo "Job cancellation process completed."
