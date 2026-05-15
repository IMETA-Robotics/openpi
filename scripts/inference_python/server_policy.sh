#!/bin/bash
gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

uv run scripts/serve_policy.py \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_rtc_inference_dual_arm \
    --policy.dir=checkpoints/pi05_base_full_dual_arm/bi_y1_test_20260410/30000
