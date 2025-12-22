#!/bin/bash
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7  # for Subscriber camera image topic

## pi0 eval
uv run scripts/inference_ros1/eval_real_robot.py policy:checkpoint \
    --policy.config=pi0_base_full_fine_tuning \
    --policy.dir=checkpoints/pi0_base_full_fine_tuning/folded_orange_towel_1122/30000

## pi05 eval
# uv run scripts/inference_ros1/eval_real_robot.py policy:checkpoint \
#     --policy.config=pi05_base_full_fine_tuning \
#     --policy.dir=checkpoints/pi05_base_full_fine_tuning/pick_up_oranges_and_place_to_plates/30000
