#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

## eval dataset
uv run scripts/eval_dataset.py policy:checkpoint \
    --policy.config=pi0_base_full_dual_arm \
    --policy.dir=checkpoints/pi0_base_full_dual_arm/folded_orange_towel_1122/30000