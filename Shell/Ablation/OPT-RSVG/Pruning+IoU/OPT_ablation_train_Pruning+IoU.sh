#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../../../" || exit 1

OUTPUT_DIR=${OUTPUT_DIR:-"output_opt_ablation_pruning_iou"}

CUDA_VISIBLE_DEVICES='0,1,2,3' torchrun \
  --nproc_per_node=4 \
  --log_dir=./train_logs \
  main.py \
  --dataset_file rsvg \
  --batch_size 8 \
  --binary \
  --with_box_refine \
  --num_frames 1 \
  --epochs 70 \
  --lr 1e-4 \
  --lr_backbone 5e-5 \
  --lr_text_encoder 1e-5 \
  --lr_poolout 1e-4 \
  --lr_drop 40 60 \
  --num_queries 10 \
  --output_dir $OUTPUT_DIR \
  --backbone resnet50 \
  --tokenizer_path ../Pretrain/RoBERTa-base \
  --text_encoder_path ../Pretrain/RoBERTa-base \
  --rsvg_path ../Dataset/OPT-RSVG \
  --use_pruning \
  --progressive_pruning \
  --pruning_ratios 0.6 0.5 0.45 0.38 \
  --pruning_temperature 0.05 \
  --use_adaptive_pruning \
  --adaptive_pruning_sensitivity 1.8 \
  --pruning_min_keep_ratio 0.3 \
  --use_iou_head \
  --use_improved_iou_head \
  --iou_loss_coef 1.1 \
  --enable_pruning_stats
