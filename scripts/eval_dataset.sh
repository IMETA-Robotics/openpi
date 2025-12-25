#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

## eval dataset
uv run scripts/eval_dataset.py policy:checkpoint \
    --policy.config=pi05_base_full_single_arm \
    --policy.dir=checkpoints/pi05_base_full_single_arm/pick_two_water_bottle/20000