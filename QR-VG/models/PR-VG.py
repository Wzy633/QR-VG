import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .text_guided_pruning import TextGuidedPruning, ProgressivePruning, DynamicVisualReconstruction
from .sparse_conv import SparseFeatureProcessor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .text_guided_pruning import ProgressivePruning

from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast

import copy
from einops import rearrange, repeat


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PRVG(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_frames, aux_loss=False, with_box_refine=False, two_stage=False,
                 freeze_text_encoder=False, args=None):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        use_iou_head = getattr(args, 'use_iou_head', False) if args is not None else False
        use_improved_iou = getattr(args, 'use_improved_iou_head', False) if args is not None else False
        if use_iou_head:
            if use_improved_iou:
                self.iou_embed = ImprovedIoUPredictor(hidden_dim, hidden_dim=hidden_dim, num_heads=8, dropout=0.1)
            else:
                self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        else:
            self.iou_embed = None
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if self.iou_embed is not None:
                self.iou_embed = _get_clones(self.iou_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.iou_embed is not None:
                self.iou_embed = nn.ModuleList([self.iou_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        tokenizer_path = getattr(args, 'tokenizer_path', '../Pretrain/RoBERTa-base') if args is not None else '../Pretrain/RoBERTa-base'
        text_encoder_path = getattr(args, 'text_encoder_path', '../Pretrain/RoBERTa-base') if args is not None else '../Pretrain/RoBERTa-base'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_path, use_safetensors=True, local_files_only=True, ignore_mismatched_sizes=True)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.fusion_module_text = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=hidden_dim)
        
        self.use_pruning = args.use_pruning if args is not None else False
        if self.use_pruning:
            use_two_stage = getattr(args, 'use_two_stage_pruning', False) if args is not None else False
            if args.progressive_pruning:
                use_adaptive = getattr(args, 'use_adaptive_pruning', True) if args is not None else True
                pruning_min_keep_ratio = getattr(args, 'pruning_min_keep_ratio', -1.0) if args is not None else -1.0
                adaptive_sensitivity = getattr(args, 'adaptive_pruning_sensitivity', 1.8) if args is not None else 1.8
                pruning_ratios_list = (getattr(args, 'pruning_ratios', None) or [0.3, 0.4, 0.5, 0.6])[:num_feature_levels]
                
                self.pruning_module = ProgressivePruning(
                    d_model=hidden_dim,
                    pruning_ratios=pruning_ratios_list,
                    temperature=args.pruning_temperature if args is not None else 0.07,
                    use_two_stage_pruning=use_two_stage,
                    use_adaptive_ratio=use_adaptive,
                    pruning_min_keep_ratio=pruning_min_keep_ratio,
                    adaptive_sensitivity=adaptive_sensitivity
                )
            else:
                pruning_ratio = args.pruning_ratio if args is not None else 0.5
                pruning_min_keep_ratio = getattr(args, 'pruning_min_keep_ratio', -1.0) if args is not None else -1.0
                adaptive_sensitivity = getattr(args, 'adaptive_pruning_sensitivity', 1.8) if args is not None else 1.8
                self.pruning_module = TextGuidedPruning(
                    d_model=hidden_dim,
                    pruning_ratio=pruning_ratio,
                    temperature=args.pruning_temperature if args is not None else 0.07,
                    use_two_stage_pruning=use_two_stage,
                    adaptive_sensitivity=adaptive_sensitivity,
                    pruning_min_keep_ratio=pruning_min_keep_ratio
                )
        else:
            self.pruning_module = None
        
        self.use_dvr = getattr(args, 'use_dvr', False) if args is not None else False
        if self.use_pruning and self.use_dvr:
            dvr_recovery_threshold = getattr(args, 'dvr_recovery_threshold', 0.6) if args is not None else 0.6
            dvr_neighbor_radius = getattr(args, 'dvr_neighbor_radius', 1) if args is not None else 1
            dvr_min_avg_similarity = getattr(args, 'dvr_min_avg_similarity', 0.3) if args is not None else 0.3
            self.dvr_module = DynamicVisualReconstruction(
                d_model=hidden_dim,
                neighbor_radius=dvr_neighbor_radius,
                recovery_threshold=dvr_recovery_threshold,
                use_text_guidance=True,
                min_avg_similarity=dvr_min_avg_similarity
            )
        else:
            self.dvr_module = None
        
        self.feature_enhancement_module = None
        
        self.use_sparse_conv = getattr(args, 'use_sparse_conv', False) if args is not None else False
        if self.use_pruning and self.use_sparse_conv:
            sparse_conv_layers = getattr(args, 'sparse_conv_layers', 1) if args is not None else 1
            self.sparse_conv_modules = nn.ModuleList([
                SparseFeatureProcessor(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_layers=sparse_conv_layers,
                    use_sparse=True
                ) for _ in range(num_feature_levels)
            ])
        else:
            self.sparse_conv_modules = None
        
        self.actual_pruning_ratios = []
        self._pruning_stats_enabled = getattr(args, 'enable_pruning_stats', False) if args is not None else False
        self.pruning_update_mask_eval = getattr(args, 'pruning_update_mask_eval', False) if args is not None else False
        self.enable_debug = getattr(args, 'enable_debug', False) if args is not None else False
        self.enable_verify_prints = getattr(args, 'enable_verify_prints', False) if args is not None else False

    def forward(self, samples: NestedTensor, captions, targets):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)

        features, pos = self.backbone(samples)

        b = len(captions)
        t = pos[0].shape[0] // b

        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
                pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            t = 1

        text_features = self.forward_text(captions, device=pos[0].device)

        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)
        text_word_initial_features = text_word_features

        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                         memory=src_proj_l,
                                                         memory_key_padding_mask=mask,
                                                         pos=pos_l,
                                                         query_pos=None)

            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_initial_features,
                                            memory_key_padding_mask=text_word_masks,
                                            pos=text_pos,
                                            query_pos=None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features = self.fusion_module_text(tgt=text_word_features,
                                                             memory=src,
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_l,
                                                             query_pos=None)
                src = self.fusion_module(tgt=src,
                                         memory=text_word_initial_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        pruning_masks_list = []
        if self.use_pruning and self.pruning_module is not None:
            text_features_for_pruning = text_word_features.permute(1, 0, 2)
            text_features_for_pruning = repeat(text_features_for_pruning, 'l b c -> l (b t) c', t=t)
            
            if isinstance(self.pruning_module, ProgressivePruning):
                self.pruning_module.reset_level()
            
            for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, poses)):
                n, c, h, w = src.shape
                src_flat = rearrange(src, '(b t) c h w -> (h w) (b t) c', b=b, t=t)
                mask_flat = rearrange(mask, '(b t) h w -> (b t) (h w)', b=b, t=t)
                mask_for_pruning = mask_flat.t()
                
                if isinstance(self.pruning_module, ProgressivePruning):
                    pruning_mask, _, _ = self.pruning_module(
                        src_flat, text_features_for_pruning, mask_for_pruning, 
                        return_pruning_mask=True, level=lvl
                    )
                else:
                    pruning_mask, _, _ = self.pruning_module(
                        src_flat, text_features_for_pruning, mask_for_pruning, return_pruning_mask=True
                    )
                
                if self._pruning_stats_enabled:
                    valid_mask = ~mask_for_pruning
                    for bt_idx in range(b*t):
                        valid_positions = valid_mask[:, bt_idx].sum().item()
                        if valid_positions > 0:
                            pruned_positions = (pruning_mask[:, bt_idx] & valid_mask[:, bt_idx]).sum().item()
                            actual_ratio = pruned_positions / valid_positions
                            self.actual_pruning_ratios.append({
                                'level': lvl,
                                'ratio': actual_ratio,
                                'h': h,
                                'w': w,
                                'total': valid_positions,
                                'pruned': pruned_positions
                            })
                
                pruning_mask_flat = pruning_mask.t()
                assert pruning_mask_flat.shape[1] == h * w, \
                    f"Dimension mismatch: pruning_mask_flat.shape[1]={pruning_mask_flat.shape[1]}, h*w={h*w}"
                pruning_mask_spatial = pruning_mask_flat.reshape(b*t, h, w)
                
                pruning_weight = (~pruning_mask_spatial).float().unsqueeze(1)
                src = src * pruning_weight
                
                if self.dvr_module is not None:
                    if self.enable_debug:
                        if torch.isnan(src).any() or torch.isinf(src).any():
                            print(f"ERROR: NaN/Inf detected in src BEFORE DVR at level {lvl}!")
                            print(f"  NaN count: {torch.isnan(src).sum().item()}, Inf count: {torch.isinf(src).sum().item()}")
                            print(f"  src stats: min={src.min().item():.6f}, max={src.max().item():.6f}, mean={src.mean().item():.6f}")
                        src_before_dvr = src.clone()
                    else:
                        src_before_dvr = None
                    
                    src = self.dvr_module(
                        features=src,
                        text_features=text_features_for_pruning,
                        pruning_mask=pruning_mask_spatial,
                        h=h,
                        w=w
                    )
                    
                    if self.training:
                        if torch.isnan(src).any() or torch.isinf(src).any():
                            if self.enable_debug:
                                print(f"ERROR: NaN/Inf detected in src AFTER DVR at level {lvl}!")
                            nan_inf_mask = torch.isnan(src).any(dim=1) | torch.isinf(src).any(dim=1)
                            pruning_mask_spatial = pruning_mask_spatial | nan_inf_mask
                            valid_src = torch.where(torch.isnan(src) | torch.isinf(src), torch.zeros_like(src), src)
                            recovered_mask = (valid_src.abs().sum(dim=1) > 0) & pruning_mask_spatial & ~nan_inf_mask
                            pruning_mask_spatial = pruning_mask_spatial & ~recovered_mask
                        else:
                            recovered_mask = (src.abs().sum(dim=1) > 0) & pruning_mask_spatial
                            pruning_mask_spatial = pruning_mask_spatial & ~recovered_mask
                    else:
                        recovered_mask = (src.abs().sum(dim=1) > 0) & pruning_mask_spatial
                        pruning_mask_spatial = pruning_mask_spatial & ~recovered_mask
                
                if self.sparse_conv_modules is not None:
                    if not hasattr(self, '_sparse_conv_verified'):
                        num_pruned = pruning_mask_spatial.sum().item()
                        total_positions = pruning_mask_spatial.numel()
                        actual_pruning_ratio = num_pruned / total_positions if total_positions > 0 else 0.0
                        if isinstance(self.pruning_module, ProgressivePruning):
                            base_ratio = self.pruning_module.get_pruning_ratio_for_level(lvl)
                        else:
                            base_ratio = getattr(self.pruning_module, 'pruning_ratio', 0.0)
                    if self.enable_verify_prints:
                        print(f"[VERIFY] Sparse Convolution ENABLED at level {lvl}")
                        print(f"  Base pruning ratio: {base_ratio:.2%}")
                        print(f"  Actual pruning ratio: {actual_pruning_ratio:.2%} ({num_pruned}/{total_positions} positions pruned)")
                        if abs(actual_pruning_ratio - base_ratio) > 0.01:
                            print(f"  ⚠️  Difference: {((actual_pruning_ratio - base_ratio) * 100):+.1f}% (adaptive adjustment)")
                        print(f"  Sparse conv will process {total_positions - num_pruned} non-zero positions")
                        self._sparse_conv_verified = True
                    
                    if self.enable_debug:
                        if torch.isnan(src).any() or torch.isinf(src).any():
                            print(f"ERROR: NaN/Inf detected in src BEFORE sparse_conv at level {lvl}!")
                            print(f"  NaN count: {torch.isnan(src).sum().item()}, Inf count: {torch.isinf(src).sum().item()}")
                            print(f"  src stats: min={src.min().item():.6f}, max={src.max().item():.6f}, mean={src.mean().item():.6f}")
                        src_before_sparse = src.clone()
                    else:
                        src_before_sparse = None
                    
                    src = self.sparse_conv_modules[lvl](
                        src,
                        pruning_mask=pruning_mask_spatial
                    )
                    
                    if self.enable_debug:
                        if torch.isnan(src).any() or torch.isinf(src).any():
                            print(f"ERROR: NaN/Inf detected in src AFTER sparse_conv at level {lvl}!")
                            print(f"  NaN count: {torch.isnan(src).sum().item()}, Inf count: {torch.isinf(src).sum().item()}")
                            print(f"  src stats: min={src.min().item():.6f}, max={src.max().item():.6f}, mean={src.mean().item():.6f}")
                            if src_before_sparse is not None:
                                print(f"  src_before_sparse stats: min={src_before_sparse.min().item():.6f}, max={src_before_sparse.max().item():.6f}, mean={src_before_sparse.mean().item():.6f}")
                
                srcs[lvl] = src
                
                pruning_mask_flatten = pruning_mask_spatial.flatten(1)
                pruning_masks_list.append(pruning_mask_flatten)
                
                if (not self.training) and self.pruning_update_mask_eval:
                    masks[lvl] = mask | pruning_mask_spatial
        else:
            pruning_masks_list = [None] * len(srcs)

        query_embeds = self.query_embed.weight
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        if self.enable_verify_prints and pruning_masks_list and not hasattr(self, '_sparse_attention_verified'):
            total_pruned = sum(mask.sum().item() if mask is not None else 0 for mask in pruning_masks_list)
            total_positions = sum(mask.numel() if mask is not None else 0 for mask in pruning_masks_list)
            if total_positions > 0:
                pruning_ratio = total_pruned / total_positions
                print(f"[VERIFY] Sparse Attention ENABLED")
                print(f"  Total pruning ratio: {pruning_ratio:.2%} ({total_pruned}/{total_positions} positions pruned)")
                print(f"  Pruning masks passed to Transformer: {len(pruning_masks_list)} levels")
                self._sparse_attention_verified = True
        
        if self.enable_debug:
            for i, src in enumerate(srcs):
                if torch.isnan(src).any() or torch.isinf(src).any():
                    print(f"ERROR: NaN/Inf detected in srcs[{i}] BEFORE Transformer!")
                    print(f"  NaN count: {torch.isnan(src).sum().item()}, Inf count: {torch.isinf(src).sum().item()}")
                    print(f"  src stats: min={src.min().item():.6f}, max={src.max().item():.6f}, mean={src.mean().item():.6f}")
        
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds, pruning_masks=pruning_masks_list)

        if self.enable_debug:
            if torch.isnan(hs).any() or torch.isinf(hs).any():
                print(f"ERROR: NaN/Inf detected in Transformer output hs!")
                print(f"  NaN count: {torch.isnan(hs).sum().item()}, Inf count: {torch.isinf(hs).sum().item()}")
                print(f"  hs stats: min={hs.min().item():.6f}, max={hs.max().item():.6f}, mean={hs.mean().item():.6f}")
                for lvl in range(hs.shape[0]):
                    nan_count = torch.isnan(hs[lvl]).sum().item()
                    if nan_count > 0:
                        print(f"    Level {lvl}: {nan_count} NaNs")
            
            if torch.isnan(init_reference).any() or torch.isinf(init_reference).any():
                print(f"ERROR: NaN/Inf detected in init_reference!")
                print(f"  NaN count: {torch.isnan(init_reference).sum().item()}")
                print(f"  init_reference stats: min={init_reference.min().item():.6f}, max={init_reference.max().item():.6f}")
            
            for i, ref in enumerate(inter_references):
                if torch.isnan(ref).any() or torch.isinf(ref).any():
                    print(f"ERROR: NaN/Inf detected in inter_references[{i}]!")
                    print(f"  NaN count: {torch.isnan(ref).sum().item()}")
                    print(f"  inter_references[{i}] stats: min={ref.min().item():.6f}, max={ref.max().item():.6f}")

        out = {}
        outputs_classes = []
        outputs_coords = []
        outputs_ious = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            if self.enable_debug:
                if torch.isnan(reference).any() or torch.isinf(reference).any():
                    print(f"ERROR: NaN/Inf detected in reference at level {lvl} BEFORE inverse_sigmoid!")
                    print(f"  reference stats: min={reference.min().item():.6f}, max={reference.max().item():.6f}")
            
            reference = inverse_sigmoid(reference)
            
            if self.enable_debug:
                if torch.isnan(reference).any() or torch.isinf(reference).any():
                    print(f"ERROR: NaN/Inf detected in reference at level {lvl} AFTER inverse_sigmoid!")
                    print(f"  reference stats: min={reference.min().item():.6f}, max={reference.max().item():.6f}")
            
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            
            if self.enable_debug:
                if torch.isnan(tmp).any() or torch.isinf(tmp).any():
                    print(f"ERROR: NaN/Inf detected in bbox_embed output at level {lvl}!")
                    print(f"  tmp stats: min={tmp.min().item():.6f}, max={tmp.max().item():.6f}")
                    print(f"  hs[lvl] stats: min={hs[lvl].min().item():.6f}, max={hs[lvl].max().item():.6f}")
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            min_size = 1e-6
            max_size = 0.99
            cx = outputs_coord[..., 0]
            cy = outputs_coord[..., 1]
            w = outputs_coord[..., 2].clamp(min=min_size, max=max_size)
            h = outputs_coord[..., 3].clamp(min=min_size, max=max_size)
            half_w = w * 0.5
            half_h = h * 0.5
            max_cx = torch.clamp(1.0 - half_w, min=half_w + min_size)
            max_cy = torch.clamp(1.0 - half_h, min=half_h + min_size)
            cx = cx.clamp(min=half_w, max=max_cx)
            cy = cy.clamp(min=half_h, max=max_cy)
            outputs_coord = torch.stack([cx, cy, w, h], dim=-1)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.iou_embed is not None:
                outputs_iou = self.iou_embed[lvl](hs[lvl]).sigmoid()
                outputs_ious.append(outputs_iou)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        
        if self.enable_debug:
            if torch.isnan(outputs_class).any() or torch.isinf(outputs_class).any():
                print(f"ERROR: NaN/Inf detected in outputs_class!")
                print(f"  NaN count: {torch.isnan(outputs_class).sum().item()}, Inf count: {torch.isinf(outputs_class).sum().item()}")
            
            if torch.isnan(outputs_coord).any() or torch.isinf(outputs_coord).any():
                print(f"ERROR: NaN/Inf detected in outputs_coord!")
                print(f"  NaN count: {torch.isnan(outputs_coord).sum().item()}, Inf count: {torch.isinf(outputs_coord).sum().item()}")
        
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]
        outputs_iou_stacked = None
        if self.iou_embed is not None and len(outputs_ious) > 0:
            outputs_iou_stacked = torch.stack(outputs_ious)
            outputs_iou_stacked = rearrange(outputs_iou_stacked, 'l (b t) q n -> l b t q n', b=b, t=t)
            if self.enable_debug:
                if torch.isnan(outputs_iou_stacked).any() or torch.isinf(outputs_iou_stacked).any():
                    print(f"ERROR: NaN/Inf detected in outputs_iou_stacked!")
                    print(f"  NaN count: {torch.isnan(outputs_iou_stacked).sum().item()}, Inf count: {torch.isinf(outputs_iou_stacked).sum().item()}")
            out['pred_iou'] = outputs_iou_stacked[-1]
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_iou_stacked)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou=None):
        if outputs_iou is not None:
            return [{"pred_logits": a, "pred_boxes": b, "pred_iou": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_iou[:-1])]
        else:
            return [{"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RobertaPoolout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ImprovedIoUPredictor(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=min(num_heads, max(1, self.hidden_dim // 32)),
            dropout=dropout,
            batch_first=False
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        self.mlp_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1)
        ])
        self.mlp_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim // 2)
        ])
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.mlp_layers[-1].bias, 0.5)
    
    def forward(self, x):
        """
        Args:
            x: [batch*time, num_queries, hidden_dim_in]
        Returns:
            iou_pred: [batch*time, num_queries, 1]
        """
        b_t, num_queries, hdim_in = x.shape
        x_flat = x.reshape(-1, hdim_in)
        x_proj = self.input_proj(x_flat)
        x_attn_in = x_proj.reshape(b_t, num_queries, -1).transpose(0, 1)
        if num_queries > 1:
            attn_out, _ = self.self_attn(x_attn_in, x_attn_in, x_attn_in)
            x_attn_in = self.attn_norm(x_attn_in + self.dropout(attn_out))
        x_flat = x_attn_in.transpose(0, 1).reshape(-1, self.hidden_dim)
        
        residual = x_flat
        for i, (layer, norm) in enumerate(zip(self.mlp_layers[:-1], self.mlp_norms)):
            x_flat = layer(x_flat)
            x_flat = F.relu(x_flat)
            x_flat = norm(x_flat)
            if x_flat.shape[-1] == residual.shape[-1]:
                x_flat = x_flat + residual
                residual = x_flat
            x_flat = self.dropout(x_flat)
        x_flat = self.mlp_layers[-1](x_flat)
        return x_flat.reshape(b_t, num_queries, 1)


class FeatureResizer(nn.Module):

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91
    device = torch.device(args.device)

    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = PRVG(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        args=args
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if getattr(args, 'use_iou_head', False):
        weight_dict['loss_iou'] = args.iou_loss_coef
    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if getattr(args, 'use_iou_head', False):
        losses += ['iou']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors