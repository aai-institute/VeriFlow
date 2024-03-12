#!/bin/bash
# Start TensorBoard in the background
mkdir ray_results
mkdir reports
poetry run tensorboard --logdir=ray_results --bind_all &
RAY=$(realpath ./ray_results)
poetry run python scripts/run-experiment.py --config experiments/mnist/mnist.yaml --report_dir reports --storage_path $RAY
