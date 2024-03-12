#!/bin/bash
# Start TensorBoard in the background
tensorboard --logdir=experiments --bind_all &

#!/bin/bash

# Base directory for experiments
BASE_DIR="experiments"

# Directory to store reports
REPORT_DIR="reports"

# Make sure the report directory exists
mkdir -p $REPORT_DIR

# Loop through all subdirectories in the experiments directory
for dataset in $BASE_DIR/*; do
    if [ -d "$dataset" ]; then
        # Loop through all .yaml files in the dataset directory
        for config in "$dataset"/*.yaml; do
            echo "Running experiment with config $config"
            
            # Run the experiment script with the given config and report directory
            poetry run python scripts/run_experiment.py --config "$config" --report_dir $REPORT_DIR --storage_path $BASE_DIR

            echo "Experiment with config $config completed"
        done
    fi
done

