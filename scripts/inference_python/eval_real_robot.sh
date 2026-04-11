#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}
export LD_LIBRARY_PATH="/home/imeta/miniconda3/envs/openpi/lib:$LD_LIBRARY_PATH"

uv run scripts/inference_python/eval_real_robot.py \
    # --visual \
    policy:checkpoint \
    --policy.config=pi05_base_full_dual_arm \
    --policy.dir=checkpoints/pi05_base_full_dual_arm/bi_y1_test_20260410/20000
