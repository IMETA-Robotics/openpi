#!/bin/bash
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}
export LD_LIBRARY_PATH="/home/imeta/miniconda3/envs/openpi/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  # for Subscriber camera image topic

## eval real robot
uv run scripts/inference_ros1/eval_real_robot.py policy:checkpoint \
    --policy.config=pi05_base_full_single_arm \
    --policy.dir=checkpoints/pi05_base_full_single_arm/pick_two_water_bottle/20000
