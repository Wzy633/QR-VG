import argparse
import datetime
import json
import random
import time
from pathlib import Path
from collections import namedtuple
from functools import partial

import os
import numpy as np
import torch

try:
    import torch
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_dir):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if torch_lib_dir not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_dir}:{current_ld_path}" if current_ld_path else torch_lib_dir
except Exception:
    pass

import torch_patch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
import datasets.samplers as samplers
from datasets.coco_eval import CocoEvaluator
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors
from tools.load_pretrained_weights import pre_trained_model_to_finetune
import opts

try:
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = None
    print("✅ 已禁用PIL解压缩炸弹检查")
except ImportError:
    pass

def main(args):
    os.environ["MDETR_CPU_REDUCE"] = "1"

    args.masks = False
    assert args.dataset_file in ["rsvg", "rsvg_mm", "refcoco", "refcoco+", "refcocog", "all"]

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if getattr(args, "enable_pruning_stats", False):
        model._pruning_stats_enabled = True
        model.actual_pruning_ratios = []
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_text_encoder_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
            "lr": args.lr_text_encoder,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.dataset_file != "all":
        dataset_train = build_dataset(args.dataset_file, image_set='train', args=args)
        print('trainset:', len(dataset_train))
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]
        dataset_train = torch.utils.data.ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in dataset_names]
        )

    print('trainset:', len(dataset_train))


    print("\nTrain dataset sample number: ", len(dataset_train))
    print("\n")

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    if args.pretrained_weights != None:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        checkpoint_dict = pre_trained_model_to_finetune(checkpoint, args)
        model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
        print("============================================>")


    if args.dataset_file != "all":
        dataset_names = [args.dataset_file]
    else:
        dataset_names = ["refcoco", "refcoco+", "refcocog"]

    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        return evaluator_list



    output_dir = Path(args.output_dir)
    if args.resume:
        print("Resume from {}".format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        filtered_state = {}
        skipped_keys = []
        
        for k, v in checkpoint_state_dict.items():
            mapped_key = k
            if k.startswith('cba_module.'):
                mapped_key = k.replace('cba_module.', 'dvr_module.', 1)
            
            if mapped_key in model_state_dict:
                if model_state_dict[mapped_key].shape == v.shape:
                    filtered_state[mapped_key] = v
                else:
                    skipped_keys.append(f"{k} -> {mapped_key}: checkpoint shape {v.shape} != model shape {model_state_dict[mapped_key].shape}")
            else:
                if k.startswith('cba_module.'):
                    pass
                else:
                    pass
        
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(filtered_state, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if len(skipped_keys) > 0:
            print('⚠️  跳过形状不匹配的键（使用默认初始化）:')
            for key in skipped_keys[:10]:
                print(f'  - {key}')
            if len(skipped_keys) > 10:
                print(f'  ... 还有 {len(skipped_keys) - 10} 个键被跳过')
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if getattr(args, "enable_pruning_stats", False):
            model.actual_pruning_ratios = []
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if getattr(args, "enable_pruning_stats", False) and len(model.actual_pruning_ratios) > 0:
            level_stats = {}
            for stat in model.actual_pruning_ratios:
                lvl = stat["level"]
                pruned = stat["pruned"]
                total = stat["total"]
                if lvl not in level_stats:
                    level_stats[lvl] = {"pruned": 0, "total": 0}
                level_stats[lvl]["pruned"] += pruned
                level_stats[lvl]["total"] += total
            pruning_summary = {}
            for lvl, v in level_stats.items():
                if v["total"] > 0:
                    pruning_summary[f"level_{lvl}_ratio"] = v["pruned"] / v["total"]
            log_stats["pruning_summary"] = pruning_summary

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer pretrain training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

