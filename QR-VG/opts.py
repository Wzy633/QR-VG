import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ReferFormer training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_text_encoder', default=1e-5, type=float)
    parser.add_argument('--lr_text_encoder_names', default=['text_encoder'], type=str, nargs='+')
    parser.add_argument('--lr_poolout', default=1e-4, type=float)
    parser.add_argument('--lr_poolout_names', default=['poolout_module'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='Number of warmup epochs for learning rate. 0 means no warmup.')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.") 

    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    parser.add_argument('--backbone', default='resnet50', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str, 
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true', help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, 
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries") 
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    
    parser.add_argument('--use_pruning', action='store_true', help='Enable Text-Guided Structured Feature Pruning')
    parser.add_argument('--pruning_ratio', default=0.5, type=float, help='Feature pruning ratio (0.5 means keep 50%)')
    parser.add_argument('--progressive_pruning', action='store_true', help='Use progressive pruning across scales')
    parser.add_argument('--pruning_ratios', default=[0.3, 0.4, 0.5, 0.6], type=float, nargs='+',
                        help='Base pruning ratios for each scale (from coarse to fine). Used by progressive pruning.')
    parser.add_argument('--pruning_temperature', default=0.07, type=float, help='Temperature for similarity computation')
    parser.add_argument('--use_adaptive_pruning', action='store_true', default=True,
                        help='Use adaptive pruning ratio based on feature importance distribution. Default: True')
    parser.add_argument('--adaptive_pruning_sensitivity', default=1.8, type=float,
                        help='Sensitivity of adaptive pruning adjustment (1.0=default, >1.0=more sensitive/larger adjustments, <1.0=more conservative). Default: 1.8')
    parser.add_argument('--pruning_min_keep_ratio', default=-1.0, type=float,
                        help='Minimum keep ratio floor for pruning (0~1). If <0, use legacy heuristic (0.7/0.75/0.8). '
                             'Lower values allow stronger pruning.')
    parser.add_argument('--use_dvr', action='store_true', help='Enable dynamic visual reconstruction (DVR) module')
    parser.add_argument('--dvr_recovery_threshold', default=0.6, type=float, help='DVR recovery threshold (higher = more conservative)')
    parser.add_argument('--dvr_neighbor_radius', default=1, type=int, help='DVR neighbor radius for feature recovery')
    parser.add_argument('--dvr_min_avg_similarity', default=0.3, type=float, 
                        help='Minimum average text similarity threshold. If avg similarity of pruned regions < this value, DVR is disabled (for very low IoU samples)')
    parser.add_argument('--use_feature_enhancement', action='store_true', help='Enable Feature Enhancement module after pruning')
    parser.add_argument('--use_two_stage_pruning', action='store_true', help='Enable two-stage pruning. Default: False (v2 uses single-stage)')
    parser.add_argument('--pruning_update_mask_eval', action='store_true',
                        help='Eval专用：剪枝后更新padding mask并移除无效位置,减少实际计算量')
    parser.add_argument('--enable_pruning_stats', action='store_true',
                        help='启用剪枝统计（训练/评估），收集实际剪枝比例用于日志分析')
    parser.add_argument('--enable_dvr_analysis', action='store_true',
                        help='启用DVR分析模式，收集每个样本的详细信息和DVR恢复统计')
    parser.add_argument('--enable_debug', action='store_true',
                        help='启用调试模式,输出详细的NaN/Inf诊断信息')
    parser.add_argument('--enable_verify_prints', action='store_true',
                        help='启用[VERIFY]类日志输出')
    parser.add_argument('--enable_tf32', action='store_true',
                        help='启用TF32加速')
    parser.add_argument('--enable_amp', action='store_true',
                        help='启用推理AMP自动混合精度')
    parser.add_argument('--amp_dtype', default='bf16', type=str, choices=['bf16', 'fp16'],
                        help='AMP dtype')
    parser.add_argument('--use_sparse_conv', action='store_true',
                        help='启用稀疏卷积模块，在剪枝后使用稀疏卷积处理特征，进一步减少计算量')
    parser.add_argument('--sparse_conv_layers', default=1, type=int,
                        help='稀疏卷积层数')
    parser.add_argument('--use_iou_head', action='store_true', help='Enable IoU prediction head')
    parser.add_argument('--use_improved_iou_head', action='store_true', help='Use improved IoU head (self-attn + deeper MLP)')
    parser.add_argument('--iou_loss_coef', default=1.0, type=float, help='IoU loss weight')
    
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--tokenizer_path', type=str, default='../Pretrain/RoBERTa-base',
                        help="Path to the tokenizer weights (local path or huggingface model name)")
    parser.add_argument('--text_encoder_path', type=str, default='../Pretrain/RoBERTa-base',
                        help="Path to the text encoder weights (local path or huggingface model name)")

    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int, 
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int, 
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int, 
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")
    
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--dataset_file', default='rsvg', help='Dataset name')
    parser.add_argument('--rsvg_path', type=str, default='../Dataset/DIOR_RSVG')
    parser.add_argument('--rsvg_mm_path', type=str, default='../Dataset/rsvg_mm')
    parser.add_argument('--rsvg_hr_path', type=str, default='data/RSVG-HR')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos')
    parser.add_argument('--davis_path', type=str, default='data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='data/a2d_sentences')
    parser.add_argument('--jhmdb_path', type=str, default='data/jhmdb_sentences')
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--ngpu', default=8, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='test', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='experiment name for saving results')
    parser.add_argument('--warmup_before_test', action='store_true', help='Warmup model before testing (skip first few batches for accurate timing)')
    parser.add_argument('--num_warmup', type=int, default=10, help='Number of warmup batches (default: 10)')
    parser.add_argument('--skip_accuracy', action='store_true', help='Skip accuracy calculation, only measure inference time (for performance testing)')
    parser.add_argument('--num_test_samples', type=int, default=None, help='Number of samples to test (None means all samples). Useful for quick performance testing.')

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    return parser
