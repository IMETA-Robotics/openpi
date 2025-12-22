#!/bin/bash
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  # for Subscriber camera image topic

uv run scripts/inference_ros2/eval_real_robot.py policy:checkpoint \
    --policy.config=pi0_dual_base_aloha_full \
    --policy.dir=checkpoints/pi0_dual_base_aloha_full/folded_orange_towel/4_L20_bs64/20000