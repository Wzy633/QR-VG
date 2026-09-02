#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../../" || exit 1

CHECKPOINT_PATH="../Model/Ablation/OPT-RSVG/Pruning+DVR/checkpoint0069.pth"
EPOCH=$(echo $CHECKPOINT_PATH | grep -oP 'checkpoint\K[0-9]+' | head -1)
EXPERIMENT_NAME="opt_ablation_pruning_dvr_epoch_${EPOCH:-unknown}"

CUDA_VISIBLE_DEVICES=0 python inference_rsvg1.py \
    --resume "$CHECKPOINT_PATH" \
    --eval \
    --dataset_file rsvg \
    --rsvg_path ../Dataset/OPT-RSVG \
    --num_queries 10 \
    --with_box_refine \
    --binary \
    --freeze_text_encoder \
    --backbone resnet50 \
    --tokenizer_path ../Pretrain/RoBERTa-base \
    --text_encoder_path ../Pretrain/RoBERTa-base \
    --use_pruning \
    --progressive_pruning \
    --pruning_ratios 0.6 0.5 0.45 0.38 \
    --pruning_temperature 0.05 \
    --use_adaptive_pruning \
    --adaptive_pruning_sensitivity 1.8 \
    --use_dvr \
    --dvr_recovery_threshold 0.6 \
    --dvr_neighbor_radius 1 \
    --pruning_min_keep_ratio 0.3 \
    --batch_size 1 \
    --num_workers 4 \
    --experiment_name "$EXPERIMENT_NAME" \
    --enable_tf32
