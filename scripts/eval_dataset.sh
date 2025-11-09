#!/bin/bash
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}

uv run scripts/eval_dataset.py policy:checkpoint \
    --policy.config=pi0_dual_base_aloha_full \
    --policy.dir=checkpoints/pi0_dual_base_aloha_full/folded_orange_towel/4_L20_bs64/29999