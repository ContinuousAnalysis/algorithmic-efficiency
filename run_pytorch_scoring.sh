#!/bin/bash

#SBATCH --nodes=1 # give it a full node
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-26
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --exclusive #this will not allow other jobs to run on this cluster
#SBATCH --output=experiments/tests/updated_schedule_free/job_%A_%a.out
#SBATCH --error=experiments/tests/updated_schedule_free/job_%A_%a.err

# Usage: sbatch <this file>.sh
# This script reads config.json and launches a sbatch job using task
# arrays where each job in the array corresponds to a training run
# for a workload given a random seed and tuning trial index.
# To generate the config.json use make_job_config.py.

set -x

# Pull docker image (ATTENTION: you may want to modify this)
REPO="europe-west4-docker.pkg.dev"
IMAGE="europe-west4-docker.pkg.dev/mlcommons-algoperf/algoperf-docker-repo/algoperf_pytorch_main:latest"
yes | gcloud auth configure-docker $REPO
docker pull $IMAGE
# Job config (ATTENTION: you may want to modify this)
config_file="$HOME/algorithmic-efficiency/pytorch_scoring_config_1.json" # Replace with your config file path
LOGS_BUCKET="algoperf-runs" # replace with your bucket used for logging


# Function to read a JSON file and extract a value by key
read_json_value() {
  local json_file="$1"
  local index="$2"
  local key="$3"
  local value=$(jq -r ".[\"$index\"].$key" "$json_file")
  echo "$value"
}

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it."
    exit 1
fi

TASK="$SLURM_ARRAY_TASK_ID"
FRAMEWORK=$(read_json_value "$config_file" "$TASK" "framework")
DATASET=$(read_json_value "$config_file" "$TASK" "dataset")
SUBMISSION_PATH=$(read_json_value "$config_file" "$TASK" "submission_path")
FRAMEWORK=$(read_json_value "$config_file" "$TASK" "framework")
TUNING_SEARCH_SPACE=$(read_json_value "$config_file" "$TASK" "tuning_search_space")
EXPERIMENT_DIR=$(read_json_value "$config_file" "$TASK" "experiment_dir")
MAX_STEPS=$(read_json_value "$config_file" "$TASK" "max_steps")
RNG_SEED=$(read_json_value "$config_file" "$TASK" "rng_seed")
WORKLOAD=$(read_json_value "$config_file" "$TASK" "workload")
HPARAM_START_INDEX=$(read_json_value "$config_file" "$TASK" "hparam_start_index")
HPARAM_END_INDEX=$(read_json_value "$config_file" "$TASK" "hparam_end_index")
NUM_TUNING_TRIALS=$(read_json_value "$config_file" "$TASK" "num_tuning_trials")
TUNING_RULESET=$(read_json_value "$config_file" "$TASK" "tuning_ruleset")
MAX_GLOBAL_STEPS=$(read_json_value "$config_file" "$TASK" "max_global_steps")

docker run \
  -v /opt/data/:/data/ \
  -v $HOME/experiment_runs:/experiment_runs \
  -v $HOME/submissions_algorithms/:/algorithmic-efficiency/submissions_algorithms \
  -v $HOME/algorithmic-efficiency/docker/scripts/startup.sh:/algorithmic-efficiency/docker/scripts/startup.sh \
  --gpus all \
  --ipc=host \
  $IMAGE \
  -d $DATASET \
  -f $FRAMEWORK \
  -s $SUBMISSION_PATH \
  -w $WORKLOAD \
  -t $TUNING_SEARCH_SPACE \
  -e $EXPERIMENT_DIR \
  -c False \
  -o True \
  --rng_seed $RNG_SEED \
  --hparam_start_index $HPARAM_START_INDEX \
  --hparam_end_index $HPARAM_END_INDEX \
  --num_tuning_trials $NUM_TUNING_TRIALS \
  --tuning_ruleset $TUNING_RULESET \
  -i true \
  -r false \
  --logs_bucket $LOGS_BUCKET \
  -m $MAX_GLOBAL_STEPS

