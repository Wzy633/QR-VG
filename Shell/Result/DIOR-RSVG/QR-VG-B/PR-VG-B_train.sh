#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../../" || exit 1

OUTPUT_DIR=${OUTPUT_DIR:-"PR-VG-B"}

CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun \
  --nproc_per_node=4 \
  main.py \
  --dataset_file rsvg \
  --binary \
  --with_box_refine \
  --batch_size 8 \
  --epochs 70 \
  --lr 1e-4 \
  --lr_backbone 5e-5 \
  --lr_drop 40 60 \
  --output_dir $OUTPUT_DIR \
  --backbone resnet50 \
  --num_queries 10 \
  --position_embedding sine \
  --enc_layers 4 \
  --dec_layers 4 \
  --dim_feedforward 2048 \
  --hidden_dim 256 \
  --dropout 0.0 \
  --nheads 8 \
  --num_feature_levels 4 \
  --tokenizer_path ../Pretrain/RoBERTa-base \
  --text_encoder_path ../Pretrain/RoBERTa-base \
  --rsvg_path ../Dataset/DIOR_RSVG \
  --use_pruning \
  --progressive_pruning \
  --pruning_ratios 0.35 0.25 0.2 0.1 \
  --pruning_temperature 0.05 \
  --use_adaptive_pruning \
  --adaptive_pruning_sensitivity 1.8 \
  --use_dvr \
  --dvr_recovery_threshold 0.6 \
  --dvr_neighbor_radius 1 \
  --pruning_min_keep_ratio 0.3 \
  --use_iou_head \
  --use_improved_iou_head \
  --iou_loss_coef 1.1 \
  --enable_pruning_stats
