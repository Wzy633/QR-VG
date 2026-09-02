import torch_patch
import argparse
import json
import random
import time
import shutil
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from util.misc import AverageMeter
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from datasets import build_dataset, get_coco_api_from_dataset
import opts
from torch.utils.data import DataLoader

from tools.colormap import colormap

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

color_list = colormap()
color_list = color_list.astype('uint8').tolist()

Visualize_bbox = False
save_visualize_path_prefix = "PR-VG_analysis"
version = "test"



def main(args):
    args.masks = False
    print("Inference only supports for batch size = 1")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        device_str = args.device if isinstance(args.device, str) else str(args.device)
        if "cuda" in device_str:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if getattr(args, "enable_verify_prints", False):
        os.environ["PRVG_VERIFY"] = "1"
    else:
        os.environ.setdefault("PRVG_VERIFY", "0")

    if getattr(args, "enable_tf32", False):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not os.path.exists(save_visualize_path_prefix):
        os.makedirs(save_visualize_path_prefix)
    
    if args.visualize:
        visualize_dir = os.path.join(save_visualize_path_prefix, version)
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

    if not args.resume:
        raise ValueError('Please specify the checkpoint for inference.')
    
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    
    if 'args' in checkpoint:
        checkpoint_args = checkpoint['args']
        print('📋 从checkpoint加载训练配置参数...')
        
        if hasattr(checkpoint_args, 'use_cba') and not hasattr(checkpoint_args, 'use_dvr'):
            checkpoint_args.use_dvr = checkpoint_args.use_cba
            print('  🔄 映射: use_cba -> use_dvr')
        if hasattr(checkpoint_args, 'cba_recovery_threshold') and not hasattr(checkpoint_args, 'dvr_recovery_threshold'):
            checkpoint_args.dvr_recovery_threshold = checkpoint_args.cba_recovery_threshold
            print('  🔄 映射: cba_recovery_threshold -> dvr_recovery_threshold')
        if hasattr(checkpoint_args, 'cba_neighbor_radius') and not hasattr(checkpoint_args, 'dvr_neighbor_radius'):
            checkpoint_args.dvr_neighbor_radius = checkpoint_args.cba_neighbor_radius
            print('  🔄 映射: cba_neighbor_radius -> dvr_neighbor_radius')
        if hasattr(checkpoint_args, 'cba_min_avg_similarity') and not hasattr(checkpoint_args, 'dvr_min_avg_similarity'):
            checkpoint_args.dvr_min_avg_similarity = checkpoint_args.cba_min_avg_similarity
            print('  🔄 映射: cba_min_avg_similarity -> dvr_min_avg_similarity')
        
        force_params = [
            'pruning_ratios',
            'pruning_temperature',
            'adaptive_pruning_sensitivity',
        ]
        
        params_to_load = [
            'pruning_min_keep_ratio',
            'use_pruning', 'progressive_pruning', 'use_adaptive_pruning',
            'use_dvr', 'dvr_recovery_threshold',
            'dvr_neighbor_radius', 'use_iou_head', 'use_improved_iou_head',
            'iou_loss_coef', 'use_sparse_conv', 'sparse_conv_layers'
        ]
        
        default_args = opts.get_args_parser().parse_args([])
        
        for param in force_params:
            checkpoint_value = getattr(checkpoint_args, param, None)
            if checkpoint_value is not None:
                cmd_value = getattr(args, param, None)
                default_value = getattr(default_args, param, None)
                
                is_explicitly_set = False
                if isinstance(cmd_value, list) and isinstance(default_value, list):
                    is_explicitly_set = (cmd_value != default_value)
                elif cmd_value != default_value:
                    is_explicitly_set = True
                
                setattr(args, param, checkpoint_value)
                if is_explicitly_set:
                    print(f'  ⚠️  {param}: 命令行={cmd_value}, checkpoint={checkpoint_value} (强制使用checkpoint值，确保与训练一致)')
                else:
                    print(f'  ✓ {param}: {checkpoint_value} (从checkpoint加载)')
        
        for param in params_to_load:
            checkpoint_value = getattr(checkpoint_args, param, None)
            if checkpoint_value is not None:
                cmd_value = getattr(args, param, None)
                default_value = getattr(default_args, param, None)
                
                if isinstance(cmd_value, list) and isinstance(default_value, list):
                    if cmd_value == default_value:
                        setattr(args, param, checkpoint_value)
                        print(f'  ✓ {param}: {checkpoint_value} (从checkpoint加载)')
                    elif cmd_value != checkpoint_value:
                        print(f'  ⚠️  {param}: 命令行={cmd_value}, checkpoint={checkpoint_value} (使用命令行值)')
                elif cmd_value == default_value:
                    setattr(args, param, checkpoint_value)
                    print(f'  ✓ {param}: {checkpoint_value} (从checkpoint加载)')
                elif cmd_value != checkpoint_value:
                    print(f'  ⚠️  {param}: 命令行={cmd_value}, checkpoint={checkpoint_value} (使用命令行值)')
        
        print('✅ 参数加载完成')
    else:
        print('⚠️  checkpoint中未找到args，使用命令行参数')

    print('正在构建数据集...')
    test_dataset = build_dataset(args.dataset_file, image_set='test', args=args)
    print(f'✅ 数据集构建完成，共 {len(test_dataset)} 个样本')
    print('正在创建数据加载器...')
    num_workers = getattr(args, 'num_workers', 4) if hasattr(args, 'num_workers') else 4
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=num_workers,
                             persistent_workers=True if num_workers > 0 else False)
    print(f'✅ 数据加载器创建完成，共 {len(test_loader)} 个batch')

    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoint_state = checkpoint['model']
    model_state = model.state_dict()
    
    filtered_state = {}
    skipped_keys = []
    for k, v in checkpoint_state.items():
        mapped_key = k
        if k.startswith('cba_module.'):
            mapped_key = k.replace('cba_module.', 'dvr_module.', 1)
        
        if mapped_key in model_state:
            if model_state[mapped_key].shape == v.shape:
                filtered_state[mapped_key] = v
            else:
                skipped_keys.append(f'{k} -> {mapped_key}: checkpoint shape {v.shape} != model shape {model_state[mapped_key].shape}')
        else:
            if k.startswith('cba_module.'):
                pass
            else:
                pass
    
    if skipped_keys:
        print('⚠️  跳过形状不匹配的键（使用默认初始化）:')
        for key in skipped_keys[:10]:
            print(f'  - {key}')
        if len(skipped_keys) > 10:
            print(f'  ... 还有 {len(skipped_keys) - 10} 个键被跳过')
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    print('✅ 模型加载完成，开始推理...')
    evaluate(test_loader, model, args)

def evaluate(test_loader, model, args):
    batch_time = AverageMeter()
    acc5 = AverageMeter()
    acc6 = AverageMeter()
    acc7 = AverageMeter()
    acc8 = AverageMeter()
    acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter()
    union_area = AverageMeter()

    device = args.device
    model.eval()
    
    inference_times = []
    inference_times_all = []
    processed_batches = 0
    
    warmup_enabled = getattr(args, 'warmup_before_test', False)
    num_warmup = getattr(args, 'num_warmup', 10)
    if warmup_enabled:
        print(f"正在预热模型（{num_warmup} 个batch，使用真实数据）...")
        warmup_count = 0
        for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
            if warmup_count >= num_warmup:
                break
            h_resize, w_resize = img.shape[-2:]
            img = img.to(device)
            captions = targets["caption"]
            size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
            target = {"size": size}
            
            with torch.inference_mode():
                _ = model(img, captions, [target])
            
            if isinstance(device, str):
                device_obj = torch.device(device)
            else:
                device_obj = device
            torch.cuda.synchronize() if device_obj.type == 'cuda' else None
            warmup_count += 1
        print(f"✓ 预热完成（{warmup_count} 个batch）")
    
    if hasattr(model, 'use_pruning') and model.use_pruning:
        model._pruning_stats_enabled = True
        model.actual_pruning_ratios = []
    
    sample_results = []
    enable_dvr_analysis = getattr(args, 'enable_dvr_analysis', False)

    skip_accuracy = getattr(args, 'skip_accuracy', False)
    num_test_samples = getattr(args, 'num_test_samples', None)
    max_batches = num_test_samples if num_test_samples is not None else len(test_loader)
    
    if skip_accuracy:
        print(f'⚠️  跳过精度计算，只测量推理时间')
    if num_test_samples is not None:
        print(f'⚠️  只测试前 {num_test_samples} 个样本（快速性能测试）')
    
    img_list = []
    count=0
    print(f'✅ 开始处理数据，共 {max_batches} 个batch')

    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    is_cuda = device_obj.type == 'cuda'
    if is_cuda:
        torch.cuda.synchronize()
    
    use_amp = getattr(args, "enable_amp", False) and is_cuda
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16

    for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
        if num_test_samples is not None and batch_idx >= num_test_samples:
            break
        if batch_idx == 0:
            print(f'✅ 第一个batch加载完成，shape: {img.shape}')
        h_resize, w_resize = img.shape[-2:]
        img = img.to(device, non_blocking=True)
        captions = targets["caption"]
        size = torch.tensor([h_resize, w_resize], dtype=torch.int64, device=device)
        target = {"size": size}

        if batch_idx == 0:
            print(f'✅ 数据准备完成，开始模型forward（第一个batch可能较慢，请耐心等待）...')
        
        inference_start = time.perf_counter()
        
        if use_amp:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(img, captions, [target])
        else:
            with torch.inference_mode():
                outputs = model(img, captions, [target])
        
        if is_cuda:
            torch.cuda.synchronize()
        inference_end = time.perf_counter()
        inference_time = inference_end - inference_start
        processed_batches += 1
        inference_times_all.append(inference_time)
        
        if batch_idx > 0:
            inference_times.append(inference_time)
        
        batch_time.update(inference_time)
        
        if batch_idx == 0:
            print(f'✅ 第一个batch处理完成（推理时间: {inference_time:.3f} 秒）')

        if skip_accuracy:
            if (batch_idx + 1) % 50 == 0:
                print(f'[{batch_idx + 1}/{max_batches}]\tTime {batch_time.avg:.3f}')
            continue


        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()
        pred_score = pred_score.squeeze(0)
        max_score, _ = pred_score.max(-1)
        
        if "pred_iou" in outputs:
            pred_iou = outputs["pred_iou"][0]
            pred_iou = pred_iou.squeeze(0)
            pred_iou = pred_iou.squeeze(-1)
            combined_score = max_score * pred_iou
            topk = combined_score.topk(k=2 if combined_score.numel() >= 2 else 1, dim=-1).indices
            best = topk[0]
            if topk.numel() >= 2:
                second = topk[1]
                best = torch.where(pred_iou[best] < 0.1, second, best)
            max_ind = best
        else:
            _, max_ind = max_score.max(-1)
        
        if torch.is_tensor(max_ind):
            max_ind = max_ind.item()
        pred_bbox = pred_bbox[0, max_ind]

        pred_bbox = rescale_bboxes(pred_bbox.detach(), (w_resize, h_resize)).numpy()
        if isinstance(targets, list):
            targets = targets[0]
        target_boxes = targets["boxes"]
        target_boxes = target_boxes.view(-1, 4)[0]
        target_bbox = rescale_bboxes(target_boxes, (w_resize, h_resize)).numpy()

        pred_bbox[0], pred_bbox[2] = (pred_bbox[0] - dw) / ratio, (pred_bbox[2] - dw) / ratio
        pred_bbox[1], pred_bbox[3] = (pred_bbox[1] - dh) / ratio, (pred_bbox[3] - dh) / ratio
        target_bbox[0], target_bbox[2] = (target_bbox[0] - dw) / ratio, (target_bbox[2] - dw) / ratio
        target_bbox[1], target_bbox[3] = (target_bbox[1] - dh) / ratio, (target_bbox[3] - dh) / ratio

        if Visualize_bbox:
                source_img = Image.open(img_path[0]).convert('RGB')

                draw = ImageDraw.Draw(source_img)
                draw_boxes = pred_bbox.tolist()

                xmin, ymin, xmax, ymax = draw_boxes[0:4]


                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[9]), width=2)
                save_visualize_path_dir = os.path.join(save_visualize_path_prefix, version)
                if not os.path.exists(save_visualize_path_dir):
                    os.makedirs(save_visualize_path_dir)
                img_name = img_path[0].split('/')[-1]
                if img_name not in img_list:
                    img_list.append(img_name)
                else:
                    count += 1
                    img_name = str(count) + '_' + img_name
                save_visualize_path = os.path.join(save_visualize_path_dir, img_name)
                source_img.save(save_visualize_path)

        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox)
        if torch.is_tensor(interArea):
            cumInterArea = float(interArea.item()) if interArea.numel() == 1 else np.sum(interArea.cpu().numpy())
            cumUnionArea = float(unionArea.item()) if unionArea.numel() == 1 else np.sum(unionArea.cpu().numpy())
        else:
            cumInterArea = np.sum(np.array(interArea))
            cumUnionArea = np.sum(np.array(unionArea))
        if torch.is_tensor(iou):
            iou_np = iou.cpu().numpy()
        else:
            iou_np = np.array(iou)
        accu5 = float((iou_np > 0.5).sum())
        accu6 = float((iou_np > 0.6).sum())
        accu7 = float((iou_np > 0.7).sum())
        accu8 = float((iou_np > 0.8).sum())
        accu9 = float((iou_np > 0.9).sum())

        meanIoU.update(torch.mean(iou).item(), img.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)

        if enable_dvr_analysis:
            iou_value = iou.item() if torch.is_tensor(iou) else float(iou)
            sample_info = {
                "sample_idx": batch_idx,
                "img_path": img_path[0] if isinstance(img_path, list) else img_path,
                "caption": captions[0] if isinstance(captions, list) else captions,
                "iou": iou_value,
                "pred_bbox": pred_bbox.tolist(),
                "target_bbox": target_bbox.tolist(),
            }
            
            if "pred_iou" in outputs:
                sample_info["pred_iou"] = pred_iou[max_ind].item() if torch.is_tensor(pred_iou) else pred_iou[max_ind]
            if torch.is_tensor(max_score):
                sample_info["max_score"] = max_score[max_ind].item()
            else:
                sample_info["max_score"] = max_score[max_ind] if isinstance(max_score, (list, np.ndarray)) else max_score
            
            if hasattr(model, 'dvr_module') and model.dvr_module is not None:
                if hasattr(model.dvr_module, '_recovery_stats'):
                    stats = model.dvr_module._recovery_stats
                    sample_info["dvr_recovered"] = stats.get('recovered_count', 0)
                    sample_info["dvr_recovery_ratio"] = stats.get('recovery_ratio', 0.0)
                    sample_info["dvr_avg_similarity"] = stats.get('avg_similarity', 0.0)
                    sample_info["dvr_avg_weight"] = stats.get('avg_weight', 0.0)
                else:
                    sample_info["dvr_recovered"] = 0
                    sample_info["dvr_recovery_ratio"] = 0.0
                    sample_info["dvr_avg_similarity"] = 0.0
                    sample_info["dvr_avg_weight"] = 0.0
            
            sample_results.append(sample_info)

        acc5.update(accu5, img.size(0))
        acc6.update(accu6, img.size(0))
        acc7.update(accu7, img.size(0))
        acc8.update(accu8, img.size(0))
        acc9.update(accu9, img.size(0))

        if batch_idx % 50 == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.avg:.3f}\t' \
                        'acc@0.5: {acc5.avg:.4f}\t' \
                        'acc@0.6: {acc6.avg:.4f}\t' \
                        'acc@0.7: {acc7.avg:.4f}\t' \
                        'acc@0.8: {acc8.avg:.4f}\t' \
                        'acc@0.9: {acc9.avg:.4f}\t' \
                        'meanIoU: {meanIoU.avg:.4f}\t' \
                        'cumuIoU: {cumuIoU:.4f}\t' \
                .format( \
                batch_idx, len(test_loader), batch_time=batch_time, \
                acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=inter_area.sum / union_area.sum)
            print(print_str)
    
    if skip_accuracy:
        final_cumuIoU = 0.0
    else:
        if union_area.sum > 0:
            final_cumuIoU = inter_area.sum / union_area.sum
        else:
            final_cumuIoU = 0.0
    
    actual_pruning_ratios_by_level = {}
    actual_computation_reduction = 0.0
    flops_reduction = 0.0
    total_flops_original = 0.0
    total_flops_reduction = 0.0
    total_flops_remaining = 0.0
    
    if hasattr(model, 'use_pruning') and model.use_pruning and len(model.actual_pruning_ratios) > 0:
        d_model = getattr(model.transformer, 'd_model', 256) if hasattr(model, 'transformer') else 256
        n_heads = 8
        
        n_levels = getattr(model, 'num_feature_levels', 4)
        
        n_points = 4
        
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            enc_layers = model.transformer.encoder.num_layers
        else:
            enc_layers = 4
            
        dim_ffn = 2048
        
        if (hasattr(model, 'transformer') and 
            hasattr(model.transformer, 'encoder') and 
            hasattr(model.transformer.encoder, 'layers') and 
            len(model.transformer.encoder.layers) > 0):
            layer = model.transformer.encoder.layers[0]
            if hasattr(layer, 'linear1'):
                dim_ffn = layer.linear1.weight.shape[0]
        
        print(f'📊 FLOPs计算参数: d_model={d_model}, n_levels={n_levels}, n_points={n_points}, enc_layers={enc_layers}, dim_ffn={dim_ffn}')
        
        level_ratios = {}
        level_flops_original = {}
        level_flops_reduction = {}
        level_features_count = {}
        
        total_features_all_levels = {}
        for stat in model.actual_pruning_ratios:
            lvl = stat['level']
            h, w = stat['h'], stat['w']
            n_features = h * w
            
            if lvl not in level_features_count:
                level_features_count[lvl] = []
            level_features_count[lvl].append(n_features)
        
        avg_features_per_level = {}
        for lvl in level_features_count.keys():
            avg_features_per_level[lvl] = sum(level_features_count[lvl]) / len(level_features_count[lvl])
        
        total_avg_features = sum(avg_features_per_level.values())
        
        for stat in model.actual_pruning_ratios:
            lvl = stat['level']
            h, w = stat['h'], stat['w']
            n_features = h * w
            pruning_ratio = stat['ratio']
            
            if lvl not in level_ratios:
                level_ratios[lvl] = []
                level_flops_original[lvl] = []
                level_flops_reduction[lvl] = []
            
            level_ratios[lvl].append(pruning_ratio)
            
            
            flops_per_feature = (
                n_levels * n_points * d_model +
                d_model * dim_ffn + dim_ffn * d_model
            )
            
            flops_original = n_features * flops_per_feature * enc_layers
            flops_reduced = pruning_ratio * flops_original
            
            level_flops_original[lvl].append(flops_original)
            level_flops_reduction[lvl].append(flops_reduced)
        
        total_flops_original_per_sample = 0.0
        total_flops_reduction_per_sample = 0.0
        total_flops_remaining_per_sample = 0.0
        
        for lvl in sorted(level_ratios.keys()):
            ratios = level_ratios[lvl]
            flops_orig = level_flops_original[lvl]
            flops_red = level_flops_reduction[lvl]
            
            if len(flops_orig) > 0:
                avg_flops_orig_per_sample = sum(flops_orig) / len(flops_orig)
                weighted_ratio = sum(r * f for r, f in zip(ratios, flops_orig)) / sum(flops_orig)
                actual_pruning_ratios_by_level[f'level_{lvl}'] = float(weighted_ratio)
            else:
                avg_flops_orig_per_sample = 0.0
                weighted_ratio = 0.0
            
            total_flops_original_per_sample += avg_flops_orig_per_sample
            total_flops_reduction_per_sample += avg_flops_orig_per_sample * weighted_ratio
        
        total_flops_original = total_flops_original_per_sample
        total_flops_reduction = total_flops_reduction_per_sample
        
        if total_flops_original > 0:
            flops_reduction = total_flops_reduction / total_flops_original
            total_flops_remaining = total_flops_original - total_flops_reduction
            actual_computation_reduction = flops_reduction
        else:
            total_flops_remaining = 0.0
        
        all_ratios = []
        all_weights = []
        for stat in model.actual_pruning_ratios:
            all_ratios.append(stat['ratio'])
            all_weights.append(stat['h'] * stat['w'])
        feature_weighted_ratio = sum(r * w for r, w in zip(all_ratios, all_weights)) / sum(all_weights) if sum(all_weights) > 0 else 0.0
        
        baseline_flops_original = total_flops_original / (1 - flops_reduction) if flops_reduction < 1.0 else total_flops_original
        
        print(f'\n📊 实际剪枝统计:')
        print(f'   特征数量加权平均剪枝比例: {feature_weighted_ratio:.4f} ({feature_weighted_ratio*100:.2f}%)')
        print(f'   FLOPs加权平均计算量减少: {flops_reduction:.4f} ({flops_reduction*100:.2f}%)')
        print(f'   改进模型（无剪枝）FLOPs: {baseline_flops_original/1e9:.2f} GFLOPs')
        print(f'   改进模型（剪枝后）FLOPs: {total_flops_original/1e9:.2f} GFLOPs')
        print(f'   剪枝减少的FLOPs: {total_flops_reduction/1e9:.2f} GFLOPs ({flops_reduction*100:.2f}%)')
        print(f'   保留的FLOPs: {total_flops_remaining/1e9:.2f} GFLOPs ({(1-flops_reduction)*100:.2f}%)')
        for lvl in sorted(actual_pruning_ratios_by_level.keys()):
            ratio = actual_pruning_ratios_by_level[lvl]
            print(f'   {lvl} (FLOPs加权): {ratio:.4f} ({ratio*100:.2f}%)')
    
    final_str = 'acc@0.5: {acc5.avg:.4f}\t' 'acc@0.6: {acc6.avg:.4f}\t' 'acc@0.7: {acc7.avg:.4f}\t' \
                'acc@0.8: {acc8.avg:.4f}\t' 'acc@0.9: {acc9.avg:.4f}\t' \
                'meanIoU: {meanIoU.avg:.4f}\t' 'cumuIoU: {cumuIoU:.4f}\t' \
        .format(acc5=acc5, acc6=acc6, acc7=acc7, acc8=acc8, acc9=acc9, \
                meanIoU=meanIoU, cumuIoU=final_cumuIoU)
    print(final_str)
    print(version)
    
    print("\n" + "="*80)
    print("实际推理时间统计 (Wall-clock Time)")
    print("="*80)
    
    if len(inference_times) > 0:
        inference_times_ms = np.array(inference_times) * 1000
        
        mean_time_ms = np.mean(inference_times_ms)
        std_time_ms = np.std(inference_times_ms)
        median_time_ms = np.median(inference_times_ms)
        min_time_ms = np.min(inference_times_ms)
        max_time_ms = np.max(inference_times_ms)
        p95_time_ms = np.percentile(inference_times_ms, 95)
        p99_time_ms = np.percentile(inference_times_ms, 99)
        
        mean_time_sec = mean_time_ms / 1000.0
        fps = 1.0 / mean_time_sec if mean_time_sec > 0 else 0.0
        
        print(f"测量次数: {len(inference_times)} 次运行（跳过第一个batch）")
        print(f"\n推理时间统计:")
        print(f"  平均时间: {mean_time_ms:.2f} ms")
        print(f"  标准差:   {std_time_ms:.2f} ms")
        print(f"  中位数:   {median_time_ms:.2f} ms")
        print(f"  最小值:   {min_time_ms:.2f} ms")
        print(f"  最大值:   {max_time_ms:.2f} ms")
        print(f"  P95:      {p95_time_ms:.2f} ms")
        print(f"  P99:      {p99_time_ms:.2f} ms")
        print(f"  吞吐量:   {fps:.2f} FPS")
        print(f"\n总样本数(实际处理): {processed_batches}")
        print(f"总推理时间(实际累计): {sum(inference_times_all):.2f} 秒")
    else:
        print(f"平均推理时间: {batch_time.avg:.3f} 秒/batch")
        print(f"平均FPS: {1.0/batch_time.avg:.2f} FPS" if batch_time.avg > 0 else "平均FPS: N/A")
        print(f"总样本数(实际处理): {processed_batches}")
        print(f"总推理时间(实际累计): {sum(inference_times_all):.2f} 秒")
    
    print("="*80)
    
    experiment_name = getattr(args, 'experiment_name', 'baseline')
    results = {
        'experiment_name': experiment_name,
        'version': version,
        'checkpoint': args.resume if args.resume else 'N/A',
        'dataset': args.dataset_file,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'skip_accuracy': skip_accuracy,
        'num_test_samples': num_test_samples if num_test_samples is not None else len(test_loader)
    }

    try:
        results['runtime'] = {
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', None),
            'torch_version': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'timing_note': 'inference_time measures model(img, captions, [target]) wall time with cuda synchronize before/after'
        }
    except Exception:
        pass
    
    if not skip_accuracy:
        results.update({
            'acc@0.5': float(acc5.avg),
            'acc@0.6': float(acc6.avg),
            'acc@0.7': float(acc7.avg),
            'acc@0.8': float(acc8.avg),
            'acc@0.9': float(acc9.avg),
            'meanIoU': float(meanIoU.avg),
            'cumuIoU': float(final_cumuIoU)
        })
    
    if actual_computation_reduction > 0:
        results['actual_computation_reduction'] = float(actual_computation_reduction)
        results['flops_reduction'] = float(flops_reduction)
        results['actual_pruning_ratios_by_level'] = actual_pruning_ratios_by_level
        if hasattr(model, 'use_pruning') and model.use_pruning and len(model.actual_pruning_ratios) > 0:
            baseline_flops_original = total_flops_original / (1 - flops_reduction) if flops_reduction < 1.0 else total_flops_original
            
            results['flops_original_gflops'] = float(total_flops_original / 1e9)
            results['flops_reduced_gflops'] = float(total_flops_reduction / 1e9)
            results['flops_remaining_gflops'] = float(total_flops_remaining / 1e9)
            results['flops_remaining_ratio'] = float(1.0 - flops_reduction)
            results['baseline_flops_gflops'] = float(baseline_flops_original / 1e9)
    
    if len(inference_times) > 0:
        inference_times_ms = np.array(inference_times) * 1000
        mean_time_ms = float(np.mean(inference_times_ms))
        std_time_ms = float(np.std(inference_times_ms))
        median_time_ms = float(np.median(inference_times_ms))
        min_time_ms = float(np.min(inference_times_ms))
        max_time_ms = float(np.max(inference_times_ms))
        p95_time_ms = float(np.percentile(inference_times_ms, 95))
        p99_time_ms = float(np.percentile(inference_times_ms, 99))
        fps = float(1.0 / (mean_time_ms / 1000.0)) if mean_time_ms > 0 else 0.0
        
        results['inference_time'] = {
            'num_runs': len(inference_times),
            'mean_time_ms': mean_time_ms,
            'std_time_ms': std_time_ms,
            'median_time_ms': median_time_ms,
            'min_time_ms': min_time_ms,
            'max_time_ms': max_time_ms,
            'p95_time_ms': p95_time_ms,
            'p99_time_ms': p99_time_ms,
            'fps': fps,
            'avg_seconds_per_batch': float(batch_time.avg),
            'total_samples': int(processed_batches),
            'total_time_seconds': float(sum(inference_times_all)),
            'warmup_excluded_samples': 1,
            'note': 'mean_time_ms/fps computed excluding the first batch; total_time_seconds sums all measured forward times'
        }
    else:
        results['inference_time'] = {
            'avg_seconds_per_batch': float(batch_time.avg),
            'avg_fps': float(1.0 / batch_time.avg) if batch_time.avg > 0 else 0.0,
            'total_samples': int(processed_batches),
            'total_time_seconds': float(sum(inference_times_all))
        }
    
    results_file = os.path.join(save_visualize_path_prefix, f'{experiment_name}_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    results_txt_file = os.path.join(save_visualize_path_prefix, f'{experiment_name}_results.txt')
    with open(results_txt_file, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('PR-VG Evaluation Results\n')
        f.write('=' * 60 + '\n')
        f.write(f'Experiment: {results["experiment_name"]}\n')
        f.write(f'Timestamp: {results["timestamp"]}\n')
        f.write(f'Dataset: {results["dataset"]}\n')
        f.write(f'Checkpoint: {results["checkpoint"]}\n')
        f.write(f'Version: {results["version"]}\n')
        f.write(f'Skip accuracy: {results.get("skip_accuracy", False)}\n')
        f.write('-' * 60 + '\n')
        f.write('Metrics:\n')

        if not results.get("skip_accuracy", False):
            f.write(f'  acc@0.5: {results["acc@0.5"]:.4f}\n')
            f.write(f'  acc@0.6: {results["acc@0.6"]:.4f}\n')
            f.write(f'  acc@0.7: {results["acc@0.7"]:.4f}\n')
            f.write(f'  acc@0.8: {results["acc@0.8"]:.4f}\n')
            f.write(f'  acc@0.9: {results["acc@0.9"]:.4f}\n')
            f.write(f'  meanIoU: {results["meanIoU"]:.4f}\n')
            f.write(f'  cumuIoU: {results["cumuIoU"]:.4f}\n')
        else:
            f.write('  (accuracy metrics skipped)\n')

        if "inference_time" in results:
            it = results["inference_time"]
            f.write('-' * 60 + '\n')
            f.write('Inference Time:\n')
            if "mean_time_ms" in it:
                f.write(f'  mean_time_ms: {it["mean_time_ms"]:.2f}\n')
                f.write(f'  p95_time_ms:  {it["p95_time_ms"]:.2f}\n')
                f.write(f'  p99_time_ms:  {it["p99_time_ms"]:.2f}\n')
                f.write(f'  fps:          {it["fps"]:.2f}\n')
            else:
                f.write(f'  avg_seconds_per_batch: {it.get("avg_seconds_per_batch", 0.0):.4f}\n')
                f.write(f'  avg_fps:               {it.get("avg_fps", 0.0):.2f}\n')

        for k in [
            "actual_computation_reduction",
            "flops_reduction",
            "flops_remaining_ratio",
        ]:
            if k in results:
                f.write(f'{k}: {results[k]}\n')

        f.write('=' * 60 + '\n')
    
    summary_file = os.path.join(save_visualize_path_prefix, 'all_experiments_summary.json')
    all_experiments = []
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            all_experiments = json.load(f)
    
    found = False
    for i, exp in enumerate(all_experiments):
        if exp.get('experiment_name') == experiment_name:
            all_experiments[i] = results
            found = True
            break
    if not found:
        all_experiments.append(results)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, indent=4, ensure_ascii=False)
    
    print(f'\n评估结果已保存到: {results_file}')
    print(f'评估结果已保存到: {results_txt_file}')
    print(f'实验汇总已更新: {summary_file}')
    
    if enable_dvr_analysis and len(sample_results) > 0:
        sample_results_file = os.path.join(save_visualize_path_prefix, f'{experiment_name}_sample_results.json')
        with open(sample_results_file, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        print(f'样本级结果已保存到: {sample_results_file}')
        print(f'  共收集 {len(sample_results)} 个样本的详细信息')
    
    checkpoint_path = args.resume if args.resume else 'N/A'
    organize_results_files(experiment_name, save_visualize_path_prefix, results_file, results_txt_file, checkpoint_path)




def organize_results_files(experiment_name, base_dir, json_file, txt_file, checkpoint_path=None):
    """
    根据实验名称自动整理结果文件到对应文件夹
    例如：pruning_smart_* -> pruning_smart_all_results/
         pruning_progressive_* -> pruning_progressive_all_results/
         iou_epoch_* -> 根据checkpoint输出目录拆分结果
    """
    
    if experiment_name.startswith('iou_epoch_'):
        if checkpoint_path and checkpoint_path != 'N/A':
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                dir_name = os.path.basename(checkpoint_dir)
                target_folder = f'{dir_name}_all_results'
            else:
                target_folder = 'iou_results'
        else:
            target_folder = 'iou_results'
    elif experiment_name.startswith('pruning_progressive_iou_optimized2'):
        target_folder = 'pruning_progressive_iou_optimized2_all_results'
    elif experiment_name.startswith('pruning_progressive_iou_optimized'):
        target_folder = 'pruning_progressive_iou_optimized_all_results'
    elif experiment_name.startswith('v6_exp'):
        if '_epoch_' in experiment_name:
            exp_base = experiment_name.split('_epoch_')[0]
            target_folder = f'{exp_base}_all_results'
        else:
            target_folder = f'{experiment_name}_all_results'
    elif experiment_name.startswith('pruning_smart'):
        target_folder = 'pruning_smart_all_results'
    elif experiment_name.startswith('pruning_progressive_v5'):
        target_folder = 'pruning_progressive_v5_all_results'
    elif experiment_name.startswith('pruning_progressive_v4_dvr'):
        target_folder = 'pruning_progressive_v4_dvr_all_results'
    elif experiment_name.startswith('pruning_progressive_v3_dvr'):
        target_folder = 'pruning_progressive_v3_dvr_all_results'
    elif experiment_name.startswith('pruning_progressive_v2'):
        target_folder = 'pruning_progressive_v2_all_results'
    elif experiment_name.startswith('pruning_progressive'):
        target_folder = 'pruning_progressive_all_results'
    elif experiment_name.startswith('pruning_ratio'):
        target_folder = 'pruning_ratio_results'
    else:
        return
    
    target_dir = os.path.join(base_dir, target_folder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f'创建文件夹: {target_dir}')
    
    files_to_move = []
    if os.path.exists(json_file):
        files_to_move.append(json_file)
    if os.path.exists(txt_file):
        files_to_move.append(txt_file)
    
    moved_count = 0
    for file_path in files_to_move:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.move(file_path, target_path)
                moved_count += 1
            except Exception as e:
                print(f'警告: 移动文件 {filename} 时出错: {e}')
    
    if moved_count > 0:
        print(f'✅ 已自动整理 {moved_count} 个结果文件到: {target_dir}/')


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])


    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return (inter_area + 1e-6) / (union_area + 1e-6), inter_area, union_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=0)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    b = (b * scale).cpu()
    return b


def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Refer_RSVG inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
