#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../../" || exit 1

CHECKPOINT_PATH="../Model/Ablation/DIOR-RSVG/Pruning+IoU/checkpoint0049.pth"
EPOCH=$(echo $CHECKPOINT_PATH | grep -oP 'checkpoint\K[0-9]+' | head -1)
EXPERIMENT_NAME="dior_ablation_pruning_iou_epoch_${EPOCH:-unknown}"

CUDA_VISIBLE_DEVICES=0 python inference_rsvg1.py \
    --resume "$CHECKPOINT_PATH" \
    --eval \
    --dataset_file rsvg \
    --rsvg_path ../Dataset/DIOR_RSVG \
    --num_queries 10 \
    --with_box_refine \
    --binary \
    --freeze_text_encoder \
    --backbone resnet50 \
    --tokenizer_path ../Pretrain/RoBERTa-base \
    --text_encoder_path ../Pretrain/RoBERTa-base \
    --use_pruning \
    --progressive_pruning \
    --pruning_ratios 0.45 0.35 0.3 0.18 \
    --pruning_temperature 0.05 \
    --use_adaptive_pruning \
    --pruning_min_keep_ratio 0.3 \
    --use_iou_head \
    --use_improved_iou_head \
    --iou_loss_coef 1.1 \
    --batch_size 1 \
    --num_workers 4 \
    --experiment_name "$EXPERIMENT_NAME" \
    --enable_tf32
