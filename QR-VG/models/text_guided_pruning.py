import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from einops import rearrange
import os


class TextGuidedPruning(nn.Module):
    
    def __init__(self, 
                 d_model: int = 256,
                 pruning_ratio: float = 0.5,
                 temperature: float = 0.07,
                 use_learnable_threshold: bool = True,
                 use_two_stage_pruning: bool = False,
                 pruning_min_keep_ratio: float = -1.0,
                 adaptive_sensitivity: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.pruning_ratio = pruning_ratio
        self.temperature = temperature
        self.use_two_stage_pruning = use_two_stage_pruning
        self.pruning_min_keep_ratio = pruning_min_keep_ratio
        self.adaptive_sensitivity = adaptive_sensitivity
        
        num_heads = min(8, d_model // 32)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, num_heads),
            batch_first=False,
            dropout=0.1,
            bias=True
        )
        
        self.text_proj = nn.Linear(d_model, d_model)
        self.vision_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.vision_proj.weight)
        if self.text_proj.bias is not None:
            nn.init.constant_(self.text_proj.bias, 0.0)
        if self.vision_proj.bias is not None:
            nn.init.constant_(self.vision_proj.bias, 0.0)
        
        if use_learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('threshold', torch.tensor(pruning_ratio))
        
        self.importance_weight = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        for m in self.importance_weight.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        import math
        self.alpha_coarse = nn.Parameter(torch.tensor(math.log(0.72 / (1 - 0.72))))
        self.alpha_fine = nn.Parameter(torch.tensor(math.log(0.82 / (1 - 0.82))))
        
        self.k_weight = nn.Parameter(torch.tensor(math.log(0.4 / (1 - 0.4))))
    
    def compute_text_vision_similarity(self, 
                                      vision_features: torch.Tensor,
                                      text_features: torch.Tensor) -> torch.Tensor:
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        text_pooled = text_proj.mean(dim=0)
        
        vision_norm = F.normalize(vision_proj, p=2, dim=-1)
        text_norm = F.normalize(text_pooled, p=2, dim=-1)
        
        similarity = torch.matmul(vision_norm, text_norm.t()) / self.temperature
        
        return similarity
    
    def compute_importance_scores(self, 
                                  vision_features: torch.Tensor,
                                  text_features: torch.Tensor,
                                  level: int = None) -> torch.Tensor:
        N, B, C = vision_features.shape
        L = text_features.shape[0]
        
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)

        use_legacy_loop = os.environ.get("PRVG_IMPORTANCE_LEGACY_LOOP", "0") == "1"
        if use_legacy_loop:
            importance_list = []
            for b in range(B):
                vision_b = vision_proj[:, b, :]
                text_b = text_proj[:, b, :]
                vision_seq = vision_b.unsqueeze(1)
                text_seq = text_b.unsqueeze(1)
                _, attn_weights = self.cross_attn(
                    query=vision_seq,
                    key=text_seq,
                    value=text_seq,
                    need_weights=True
                )
                attn_weights = attn_weights.squeeze(0)

                L_cur = attn_weights.shape[1]
                attn_variance = attn_weights.var(dim=1).mean()
                k_weight_sigmoid = torch.sigmoid(self.k_weight)
                if L_cur <= 3:
                    k_top = L_cur
                else:
                    variance_factor = 1.0 + attn_variance * 0.5
                    base_k_ratio_tensor = k_weight_sigmoid * variance_factor
                    base_k_ratio = float(base_k_ratio_tensor.clamp(0.25, 0.6).item())
                    k_base = max(1, int(L_cur * base_k_ratio))
                    if L_cur <= 10:
                        k_max = max(1, min(8, L_cur // 2))
                    elif L_cur <= 20:
                        k_max = max(1, min(12, L_cur // 2))
                    else:
                        k_max = max(1, min(16, L_cur // 2))
                    k_min = max(1, int(L_cur * 0.25))
                    k_top = max(k_min, min(k_max, k_base, L_cur))

                topk_values, _ = torch.topk(attn_weights, k=k_top, dim=1)
                weights = F.softmax(topk_values, dim=1)
                similarity_score = (topk_values * weights).sum(dim=1)

                vision_b_flat = vision_features[:, b, :]
                feature_importance = self.importance_weight(vision_b_flat).squeeze(-1)

                sim_min, sim_max = similarity_score.min(), similarity_score.max()
                imp_min, imp_max = feature_importance.min(), feature_importance.max()
                sim_range = sim_max - sim_min
                imp_range = imp_max - imp_min
                if sim_range < 1e-5:
                    similarity_norm = torch.zeros_like(similarity_score)
                else:
                    similarity_norm = (similarity_score - sim_min) / (sim_range + 1e-8)
                if imp_range < 1e-5:
                    importance_norm = torch.zeros_like(feature_importance)
                else:
                    importance_norm = (feature_importance - imp_min) / (imp_range + 1e-8)

                if level is not None:
                    if level <= 1:
                        alpha = torch.sigmoid(self.alpha_coarse)
                    else:
                        alpha = torch.sigmoid(self.alpha_fine)
                else:
                    alpha = (torch.sigmoid(self.alpha_coarse) + torch.sigmoid(self.alpha_fine)) / 2.0
                importance_b = alpha * similarity_norm + (1 - alpha) * importance_norm
                importance_list.append(importance_b)

            return torch.stack(importance_list, dim=1)
        
        _, attn_weights_all = self.cross_attn(
            query=vision_proj,
            key=text_proj,
            value=text_proj,
            need_weights=True
        )
        
        importance_list = []
        for b in range(B):
            attn_weights = attn_weights_all[b]
            
            L_cur = attn_weights.shape[1]
            
            attn_variance = attn_weights.var(dim=1).mean()
            
            k_weight_sigmoid = torch.sigmoid(self.k_weight)
            
            if L_cur <= 3:
                k_top = L_cur
            else:
                variance_factor = 1.0 + attn_variance * 0.5
                base_k_ratio_tensor = k_weight_sigmoid * variance_factor
                base_k_ratio = float(base_k_ratio_tensor.clamp(0.25, 0.6).item())
                k_base = max(1, int(L_cur * base_k_ratio))
                
                if L_cur <= 10:
                    k_max = max(1, min(8, L_cur // 2))
                elif L_cur <= 20:
                    k_max = max(1, min(12, L_cur // 2))
                else:
                    k_max = max(1, min(16, L_cur // 2))
                k_min = max(1, int(L_cur * 0.25))
                k_top = max(k_min, min(k_max, k_base, L_cur))
            
            topk_values, _ = torch.topk(attn_weights, k=k_top, dim=1)
            weights = F.softmax(topk_values, dim=1)
            similarity_score = (topk_values * weights).sum(dim=1)
            
            vision_b_flat = vision_features[:, b, :]
            feature_importance = self.importance_weight(vision_b_flat).squeeze(-1)
            
            sim_min, sim_max = similarity_score.min(), similarity_score.max()
            imp_min, imp_max = feature_importance.min(), feature_importance.max()
            sim_range = sim_max - sim_min
            imp_range = imp_max - imp_min
            similarity_norm = (similarity_score - sim_min) / sim_range if sim_range > 0 else torch.zeros_like(similarity_score)
            importance_norm = (feature_importance - imp_min) / imp_range if imp_range > 0 else torch.zeros_like(feature_importance)
            
            if level is not None:
                if level <= 1:
                    alpha = torch.sigmoid(self.alpha_coarse)
                else:
                    alpha = torch.sigmoid(self.alpha_fine)
            else:
                alpha = (torch.sigmoid(self.alpha_coarse) + torch.sigmoid(self.alpha_fine)) / 2.0
            importance_b = alpha * similarity_norm + (1 - alpha) * importance_norm
            importance_list.append(importance_b)
        
        importance = torch.stack(importance_list, dim=1)
        
        return importance
    
    def compute_adaptive_pruning_ratio(self, 
                                       importance: torch.Tensor,
                                       mask: torch.Tensor = None,
                                       level: int = None,
                                       base_ratio: float = None) -> float:
        if base_ratio is None:
            base_ratio = self.pruning_ratio
        
        N, B = importance.shape
        
        if mask is not None:
            valid_importance = importance[~mask]
        else:
            valid_importance = importance.flatten()
        
        if valid_importance.numel() == 0:
            return base_ratio
        
        if self.training:
            importance_mean = valid_importance.mean().item()
            importance_std = valid_importance.std().item()
            importance_median = valid_importance.median().item()
            
            q25 = valid_importance.quantile(0.25).item()
            q75 = valid_importance.quantile(0.75).item()
            iqr = q75 - q25
        else:
            importance_mean = valid_importance.mean()
            importance_std = valid_importance.std()
            importance_median = valid_importance.median()
            q25 = valid_importance.quantile(0.25)
            q75 = valid_importance.quantile(0.75)
            iqr = q75 - q25
        
        if importance_mean > 1e-6:
            cv = importance_std / importance_mean
        else:
            cv = 0.0
        
        if iqr > 1e-6:
            skewness_estimate = (importance_median - importance_mean) / (iqr + 1e-8)
        else:
            skewness_estimate = 0.0
        
        
        if cv < 0.1:
            cv_factor_base = 1.15
        elif cv < 0.2:
            cv_factor_base = 1.05
        elif cv < 0.4:
            cv_factor_base = 1.0
        else:
            cv_factor_base = 0.9
        
        cv_factor = 1.0 + (cv_factor_base - 1.0) * self.adaptive_sensitivity
        
        if skewness_estimate < -0.3:
            skew_factor_base = 1.1
        elif skewness_estimate > 0.3:
            skew_factor_base = 0.95
        else:
            skew_factor_base = 1.0
        
        skew_factor = 1.0 + (skew_factor_base - 1.0) * self.adaptive_sensitivity
        
        if level is not None:
            if level <= 1:
                level_factor_base = 1.05
            else:
                level_factor_base = 0.95
        else:
            level_factor_base = 1.0
        
        level_factor = 1.0 + (level_factor_base - 1.0) * self.adaptive_sensitivity
        
        if N < 100:
            size_factor = 0.9
        elif N < 500:
            size_factor = 0.95
        else:
            size_factor = 1.0
        
        adaptive_ratio = base_ratio * cv_factor * skew_factor * level_factor * size_factor
        
        adaptive_ratio = max(0.1, min(0.6, adaptive_ratio))
        
        return adaptive_ratio
    
    def forward(self,
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                mask: torch.Tensor = None,
                keep_ratio: float = None,
                return_pruning_mask: bool = False,
                use_adaptive_ratio: bool = True,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        N, B, C = vision_features.shape
        
        level = kwargs.get('level', None)
        
        importance = self.compute_importance_scores(vision_features, text_features, level=level)
        
        if mask is not None:
            if mask.shape == (B, N):
                mask = mask.t()
            elif mask.shape != (N, B):
                if mask.numel() == N * B:
                    mask = mask.reshape(N, B)
                else:
                    print(f"Warning: mask shape {mask.shape} doesn't match expected [N={N}, B={B}], skipping mask")
                    mask = None
            
            if mask is not None:
                importance = importance.masked_fill(mask, float('-inf'))
        
        if keep_ratio is None:
            if use_adaptive_ratio:
                base_ratio = self.pruning_ratio
                adaptive_pruning_ratio = self.compute_adaptive_pruning_ratio(
                    importance, mask, level, base_ratio
                )
                if os.environ.get("PRVG_VERIFY", "0") == "1" and not hasattr(self, '_adaptive_ratio_verified'):
                    print(f"[VERIFY] Adaptive Pruning Ratio at level {level if level is not None else 'unknown'}")
                    print(f"  Base ratio: {base_ratio:.2%}")
                    print(f"  Adaptive ratio: {adaptive_pruning_ratio:.2%}")
                    print(f"  Adjustment: {((adaptive_pruning_ratio / base_ratio - 1) * 100):+.1f}%")
                    self._adaptive_ratio_verified = True
                keep_ratio = 1.0 - adaptive_pruning_ratio
            else:
                keep_ratio = 1.0 - self.pruning_ratio
        
        if self.use_two_stage_pruning:
            pruning_mask = self._two_stage_pruning(
                importance, mask, keep_ratio, N, B
            )
        else:
            pruning_mask = self._single_stage_pruning(
                importance, mask, keep_ratio, N, B
            )
        
        if return_pruning_mask:
            return pruning_mask, mask, None
        
        pruned_features_list = []
        pruned_mask_list = []
        keep_indices_list = []
        
        for b in range(B):
            keep_mask = ~pruning_mask[:, b]
            indices = torch.where(keep_mask)[0]
            indices = torch.clamp(indices, 0, N - 1)
            indices = torch.unique(indices, sorted=True)
            k_actual = len(indices)
            
            if k_actual == 0:
                importance_b = importance[:, b]
                _, fallback_idx = torch.topk(importance_b, 1, dim=0)
                indices = torch.clamp(fallback_idx, 0, N - 1)
                k_actual = 1
            
            pruned_feat = vision_features[indices, b, :]
            pruned_features_list.append(pruned_feat)
            
            if mask is not None:
                pruned_mask_b = ~mask[indices, b]
            else:
                pruned_mask_b = torch.ones(k_actual, dtype=torch.bool, device=vision_features.device)
            pruned_mask_list.append(pruned_mask_b)
            
            keep_indices_list.append(indices)
        
        max_k = max(len(indices) for indices in keep_indices_list)
        
        if all(len(indices) == max_k for indices in keep_indices_list):
            pruned_features = torch.stack(pruned_features_list, dim=1)
            pruned_mask = torch.stack(pruned_mask_list, dim=0)
        else:
            padded_features = []
            padded_masks = []
            for feat, msk in zip(pruned_features_list, pruned_mask_list):
                pad_size = max_k - feat.shape[0]
                if pad_size > 0:
                    feat_padded = F.pad(feat, (0, 0, 0, pad_size), value=0)
                    msk_padded = F.pad(msk.float(), (0, pad_size), value=0).bool()
                else:
                    feat_padded = feat
                    msk_padded = msk
                padded_features.append(feat_padded)
                padded_masks.append(msk_padded)
            pruned_features = torch.stack(padded_features, dim=1)
            pruned_mask = torch.stack(padded_masks, dim=0)
        
        return pruned_features, pruned_mask, keep_indices_list
    
    def _two_stage_pruning(self, importance, mask, keep_ratio, N, B):
        pruning_mask = torch.zeros(N, B, dtype=torch.bool, device=importance.device)
        
        for b in range(B):
            importance_b = importance[:, b]
            
            if mask is not None:
                valid_mask = ~mask[:, b]
                importance_b = importance_b.clone()
                importance_b[~valid_mask] = float('-inf')
            else:
                valid_mask = torch.ones(N, dtype=torch.bool, device=importance_b.device)
            
            valid_importance = importance_b[valid_mask] if valid_mask.any() else importance_b
            
            if valid_importance.numel() == 0 or valid_importance.max() <= float('-inf'):
                continue
            
            stage1_multiplier = 1.05
            max_stage1_ratio = min(self.pruning_ratio * 1.2, 0.33)
            scene_pruning_ratio = min(max_stage1_ratio, self.pruning_ratio * stage1_multiplier)
            scene_threshold = torch.quantile(valid_importance, scene_pruning_ratio)
            scene_keep_mask = importance_b >= scene_threshold
            
            min_keep_scene = max(1, int(N * (1 - scene_pruning_ratio) * 0.85))
            if scene_keep_mask.sum() < min_keep_scene:
                _, topk_indices = torch.topk(importance_b, min_keep_scene, dim=0)
                scene_keep_mask = torch.zeros_like(importance_b, dtype=torch.bool)
                scene_keep_mask[topk_indices] = True
            
            kept_indices = torch.where(scene_keep_mask)[0]
            if len(kept_indices) == 0:
                continue
            
            kept_importance = importance_b[kept_indices]
            
            target_keep = max(1, int(N * keep_ratio))
            if len(kept_indices) <= target_keep:
                final_keep_mask = scene_keep_mask
            else:
                stage2_keep_num = target_keep
                _, topk_indices_stage2 = torch.topk(kept_importance, stage2_keep_num, dim=0)
                final_kept_indices = kept_indices[topk_indices_stage2]
                
                final_keep_mask = torch.zeros_like(importance_b, dtype=torch.bool)
                final_keep_mask[final_kept_indices] = True
            
            final_keep_mask = final_keep_mask | (~valid_mask)
            pruning_mask[:, b] = ~final_keep_mask
        
        return pruning_mask
    
    def _single_stage_pruning(self, importance, mask, keep_ratio, N, B):
        pruning_mask = torch.zeros(N, B, dtype=torch.bool, device=importance.device)
        
        for b in range(B):
            importance_b = importance[:, b]
            
            if mask is not None:
                valid_mask = ~mask[:, b]
                importance_b = importance_b.clone()
                importance_b[~valid_mask] = float('-inf')
                valid_importance = importance_b[valid_mask] if valid_mask.any() else importance_b
            else:
                valid_mask = torch.ones(N, dtype=torch.bool, device=importance_b.device)
                valid_importance = importance_b
            
            if valid_importance.numel() > 0 and valid_importance.max() > float('-inf'):
                quantile_threshold = torch.quantile(valid_importance, self.pruning_ratio)
                
                importance_mean = valid_importance.mean()
                importance_std = valid_importance.std()
                importance_median = valid_importance.median()
                
                q25 = valid_importance.quantile(0.25)
                q75 = valid_importance.quantile(0.75)
                iqr = q75 - q25
                skewness = (importance_median - importance_mean) / iqr.clamp(min=1e-8)
                threshold_adjustment = -0.1 * skewness
                adjusted_threshold = quantile_threshold + threshold_adjustment * importance_std
                threshold_min = valid_importance.min()
                threshold_max = valid_importance.max()
                adjusted_threshold = torch.clamp(adjusted_threshold, threshold_min, threshold_max)
                
                keep_mask = importance_b >= adjusted_threshold
                
                if self.pruning_min_keep_ratio is not None and self.pruning_min_keep_ratio >= 0:
                    floor = float(self.pruning_min_keep_ratio)
                    floor = max(0.0, min(1.0, floor))
                    min_keep_ratio = max(floor, keep_ratio)
                else:
                    if self.pruning_ratio < 0.2:
                        min_keep_ratio = max(0.8, keep_ratio)
                    elif self.pruning_ratio < 0.3:
                        min_keep_ratio = max(0.75, keep_ratio)
                    else:
                        min_keep_ratio = max(0.7, keep_ratio)
                min_keep = max(1, int(N * min_keep_ratio))
                if keep_mask.sum() < min_keep:
                    _, topk_indices = torch.topk(importance_b, min_keep, dim=0)
                    keep_mask = torch.zeros_like(importance_b, dtype=torch.bool)
                    keep_mask[topk_indices] = True
                
                keep_mask = keep_mask | (~valid_mask)
            else:
                keep_mask = torch.ones(N, dtype=torch.bool, device=importance_b.device)
            
            pruning_mask[:, b] = ~keep_mask
        
        return pruning_mask


class MultiScaleFeatureInteraction(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=False,
            dropout=0.1
        )
        self.scale_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.scale_proj.weight)
        if self.scale_proj.bias is not None:
            nn.init.constant_(self.scale_proj.bias, 0.0)
    
    def forward(self, current_features, reference_features, reference_importance):
        ref_weights = reference_importance.unsqueeze(-1)
        
        ref_proj = self.scale_proj(reference_features)
        curr_proj = self.scale_proj(current_features)
        
        weighted_ref = ref_proj * ref_weights
        
        attn_output, _ = self.cross_scale_attn(
            query=curr_proj,
            key=ref_proj,
            value=weighted_ref,
            need_weights=False
        )
        
        enhanced_features = current_features + 0.3 * attn_output
        
        del ref_proj, curr_proj, weighted_ref, attn_output
        
        return enhanced_features


class ProgressivePruning(nn.Module):
    
    def __init__(self,
                 d_model: int = 256,
                 pruning_ratios: List[float] = [0.3, 0.4, 0.5, 0.6],
                 temperature: float = 0.07,
                 use_two_stage_pruning: bool = False,
                 use_adaptive_ratio: bool = True,
                 pruning_min_keep_ratio: float = -1.0,
                 adaptive_sensitivity: float = 1.0):
        super().__init__()
        self.use_adaptive_ratio = use_adaptive_ratio
        self.pruning_min_keep_ratio = pruning_min_keep_ratio
        self.adaptive_sensitivity = adaptive_sensitivity
        
        self.num_levels = len(pruning_ratios)
        self.pruning_ratios = pruning_ratios
        
        if isinstance(temperature, (list, tuple)) and len(temperature) == self.num_levels:
            temperatures = temperature
        else:
            base_temp = temperature if isinstance(temperature, (float, int)) else 0.07
            temperatures = [base_temp + 0.01 * (self.num_levels - 1 - i) for i in range(self.num_levels)]
        
        self.pruners = nn.ModuleList([
            TextGuidedPruning(
                d_model=d_model,
                pruning_ratio=ratio,
                temperature=temp,
                use_two_stage_pruning=use_two_stage_pruning,
                pruning_min_keep_ratio=pruning_min_keep_ratio,
                adaptive_sensitivity=adaptive_sensitivity
            ) for ratio, temp in zip(self.pruning_ratios, temperatures)
        ])
        
        
        self.temperatures = temperatures
        
        self.current_level = 0
        self.previous_level_features = None
        self.previous_level_importance = None
    
    def reset_level(self):
        self.current_level = 0
        self.previous_level_features = None
        self.previous_level_importance = None

    def get_pruning_ratio_for_level(self, level: int) -> float:
        return float(self.pruning_ratios[min(level, len(self.pruning_ratios) - 1)])
    
    def forward(self,
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                mask: torch.Tensor = None,
                keep_ratio: float = None,
                return_pruning_mask: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        level = kwargs.get('level', None)
        if level is None:
            level = self.current_level
            self.current_level = (self.current_level + 1) % self.num_levels
        else:
            level = min(level, self.num_levels - 1)
        
        
        pruner = self.pruners[level]

        pruner.pruning_ratio = self.get_pruning_ratio_for_level(level)
        
        pruner._current_level = level
        
        result = pruner(
            vision_features,
            text_features,
            mask,
            keep_ratio,
            return_pruning_mask,
            level=level,
            use_adaptive_ratio=self.use_adaptive_ratio
        )
        
        if return_pruning_mask:
            pruning_mask, original_mask, _ = result
            importance = pruner.compute_importance_scores(vision_features, text_features, level=level)
            if mask is not None:
                importance = importance.masked_fill(mask, float('-inf'))
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            self.previous_level_features = vision_features.detach()
            self.previous_level_importance = importance.detach()
        else:
            pruned_features, pruned_mask, _ = result
            self.previous_level_features = pruned_features.detach()
            importance = torch.ones(pruned_features.shape[0], pruned_features.shape[1], device=pruned_features.device)
            self.previous_level_importance = importance.detach()
        
        return result
    
    def forward_batch(self,
                     multi_scale_features: List[torch.Tensor],
                     text_features: torch.Tensor,
                     masks: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pruned_features = []
        pruned_masks = []
        
        if masks is None:
            masks = [None] * len(multi_scale_features)
        
        for i, (feat, mask, pruner) in enumerate(zip(multi_scale_features, masks, self.pruners)):
            pruned_feat, pruned_mask, _ = pruner(feat, text_features, mask)
            pruned_features.append(pruned_feat)
            pruned_masks.append(pruned_mask)
        
        return pruned_features, pruned_masks


class DynamicVisualReconstruction(nn.Module):
    
    def __init__(self,
                 d_model: int = 256,
                 neighbor_radius: int = 1,
                 recovery_threshold: float = 0.6,
                 use_text_guidance: bool = True,
                 min_avg_similarity: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.neighbor_radius = neighbor_radius
        self.use_text_guidance = use_text_guidance
        self.min_avg_similarity = min_avg_similarity
        
        import math
        norm_threshold = (recovery_threshold - 0.2) / 0.5
        norm_threshold = max(0.0, min(1.0, norm_threshold))
        self.recovery_threshold_param = nn.Parameter(torch.tensor(math.log(norm_threshold / (1 - norm_threshold + 1e-8) + 1e-8)))
        
        
        self.completion_net_main = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model * 2, d_model)
        )
        
        
        if use_text_guidance:
            self.text_proj = nn.Linear(d_model, d_model)
            self.vision_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.xavier_uniform_(self.vision_proj.weight)
            if self.text_proj.bias is not None:
                nn.init.constant_(self.text_proj.bias, 0.0)
            if self.vision_proj.bias is not None:
                nn.init.constant_(self.vision_proj.bias, 0.0)
        
        for m in self.completion_net_main.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
    
    def compute_neighbor_features(self,
                                 features: torch.Tensor,
                                 pruning_mask: torch.Tensor,
                                 h: int,
                                 w: int) -> torch.Tensor:
        B_T, C, H, W = features.shape
        device = features.device
        
        neighbor_features = torch.zeros_like(features)
        
        kernel_size = 2 * self.neighbor_radius + 1
        neighbor_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
        
        kept_features = features * (~pruning_mask).float().unsqueeze(1)
        
        neighbor_features = F.conv2d(
            kept_features.view(B_T * C, 1, H, W),
            neighbor_kernel,
            padding=self.neighbor_radius
        ).view(B_T, C, H, W)
        
        neighbor_features = neighbor_features * pruning_mask.float().unsqueeze(1)
        
        return neighbor_features
    
    def compute_text_similarity(self,
                               features: torch.Tensor,
                               text_features: torch.Tensor,
                               pruning_mask: torch.Tensor,
                               neighbor_features: torch.Tensor = None) -> torch.Tensor:
        if not self.use_text_guidance:
            return torch.zeros(features.shape[0], features.shape[2], features.shape[3], 
                             device=features.device)
        
        B_T, C, H, W = features.shape
        L = text_features.shape[0]
        
        
        text_proj = self.text_proj(text_features)
        if torch.isnan(text_proj).any() or torch.isinf(text_proj).any():
            text_proj = torch.where(torch.isnan(text_proj) | torch.isinf(text_proj),
                                   torch.zeros_like(text_proj), text_proj)
        
        text_pooled = text_proj.mean(dim=0)
        if self.training:
            if torch.isnan(text_pooled).any() or torch.isinf(text_pooled).any():
                text_pooled = torch.where(torch.isnan(text_pooled) | torch.isinf(text_pooled),
                                         torch.zeros_like(text_pooled), text_pooled)
        
        text_norm_sum = text_pooled.norm(p=2, dim=-1, keepdim=True)
        text_norm = text_pooled / text_norm_sum.clamp(min=1e-8)
        
        text_norm_expanded = text_norm.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1).permute(0, 3, 1, 2)
        text_norm_expanded_sum = text_norm_expanded.norm(p=2, dim=1, keepdim=True)
        text_norm_expanded = text_norm_expanded / text_norm_expanded_sum.clamp(min=1e-8)
        
        features_proj = self.vision_proj(features)
        if self.training:
            if torch.isnan(features_proj).any() or torch.isinf(features_proj).any():
                features_proj = torch.where(torch.isnan(features_proj) | torch.isinf(features_proj),
                                           torch.zeros_like(features_proj), features_proj)
        
        pruning_mask_expanded = pruning_mask.unsqueeze(1).float()
        
        if neighbor_features is not None:
            neighbor_proj = self.vision_proj(neighbor_features)
            if self.training:
                if torch.isnan(neighbor_proj).any() or torch.isinf(neighbor_proj).any():
                    neighbor_proj = torch.where(torch.isnan(neighbor_proj) | torch.isinf(neighbor_proj),
                                               torch.zeros_like(neighbor_proj), neighbor_proj)
            
            neighbor_has_value = (neighbor_features.abs().sum(dim=1, keepdim=True) > 0)
            neighbor_mask = neighbor_has_value.float()
            
            features_for_sim = features_proj * (1 - pruning_mask_expanded) + \
                              (neighbor_proj * neighbor_mask + text_norm_expanded * (1 - neighbor_mask)) * pruning_mask_expanded
        else:
            features_proj_norm_sum = features_proj.norm(p=2, dim=1, keepdim=True)
            features_norm = features_proj / features_proj_norm_sum.clamp(min=1e-8)
            features_for_sim = features_norm * (1 - pruning_mask_expanded) + text_norm_expanded * pruning_mask_expanded
        
        features_norm_sum = features_for_sim.norm(p=2, dim=1, keepdim=True)
        features_for_sim_norm = features_for_sim / features_norm_sum.clamp(min=1e-8)
        
        similarity = (features_for_sim_norm * text_norm_expanded).sum(dim=1)
        
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            similarity = torch.where(torch.isnan(similarity) | torch.isinf(similarity),
                                     torch.zeros_like(similarity), similarity)
        
        similarity = torch.clamp(similarity, -1.0, 1.0)
        
        return similarity
    
    def forward(self,
                features: torch.Tensor,
                text_features: torch.Tensor,
                pruning_mask: torch.Tensor,
                h: int,
                w: int) -> torch.Tensor:
        B_T, C, H, W = features.shape
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            return torch.where(torch.isnan(features) | torch.isinf(features), 
                              torch.zeros_like(features), features)
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            return features
        
        if not pruning_mask.any():
            return features
        
        neighbor_features = self.compute_neighbor_features(
            features, pruning_mask, h, w
        )
        
        if torch.isnan(neighbor_features).any() or torch.isinf(neighbor_features).any():
            neighbor_features = torch.where(torch.isnan(neighbor_features) | torch.isinf(neighbor_features),
                                           torch.zeros_like(neighbor_features), neighbor_features)
        
        text_similarity = self.compute_text_similarity(
            features, text_features, pruning_mask, neighbor_features
        )
        
        pruned_similarity = text_similarity[pruning_mask]
        if pruned_similarity.numel() > 0:
            avg_pruned_similarity = pruned_similarity.mean()
            avg_pruned_similarity_value = avg_pruned_similarity.item()
        else:
            avg_pruned_similarity = torch.tensor(0.0, device=text_similarity.device)
            avg_pruned_similarity_value = 0.0
        
        if avg_pruned_similarity_value < self.min_avg_similarity:
            if self.training:
                self._recovery_stats = {
                    'recovered_count': 0,
                    'recovery_ratio': 0.0,
                    'avg_similarity': avg_pruned_similarity_value,
                    'avg_weight': 0.0,
                    'total_pruned': pruning_mask.sum().item(),
                    'dvr_disabled': True
                }
            return features
        
        neighbor_has_value = (neighbor_features.abs().sum(dim=1) > 0)
        
        recovery_threshold_sigmoid = torch.sigmoid(self.recovery_threshold_param)
        learned_threshold = 0.2 + recovery_threshold_sigmoid * 0.5
        high_similarity = text_similarity > learned_threshold
        
        should_recover = pruning_mask & neighbor_has_value & high_similarity
        
        if not should_recover.any():
            return features
        
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            text_features = torch.where(torch.isnan(text_features) | torch.isinf(text_features),
                                       torch.zeros_like(text_features), text_features)
        
        text_pooled = text_features.mean(dim=0)
        if torch.isnan(text_pooled).any() or torch.isinf(text_pooled).any():
            text_pooled = torch.where(torch.isnan(text_pooled) | torch.isinf(text_pooled),
                                     torch.zeros_like(text_pooled), text_pooled)
        
        text_expanded = text_pooled.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        completion_input = torch.cat([text_expanded, neighbor_features], dim=1)
        
        completion_input = torch.clamp(completion_input, min=-10.0, max=10.0)
        
        completion_input_flat = rearrange(completion_input, 'b c h w -> (h w) b c')
        
        has_nan_inf = torch.isnan(completion_input_flat).any() or torch.isinf(completion_input_flat).any()
        if has_nan_inf:
            nan_inf_mask = torch.isnan(completion_input_flat) | torch.isinf(completion_input_flat)
            completion_input_flat = torch.where(nan_inf_mask, 
                                              torch.zeros_like(completion_input_flat), 
                                              completion_input_flat)
        
        recovered_features_flat = self.completion_net_main(completion_input_flat)
        
        has_nan_inf_output = torch.isnan(recovered_features_flat).any() or torch.isinf(recovered_features_flat).any()
        if has_nan_inf_output:
            nan_inf_mask_output = torch.isnan(recovered_features_flat) | torch.isinf(recovered_features_flat)
            recovered_features_flat = torch.where(nan_inf_mask_output,
                                                torch.zeros_like(recovered_features_flat),
                                                recovered_features_flat)
        
        recovered_features_flat = torch.clamp(recovered_features_flat, min=-10.0, max=10.0)
        
        recovered_features = rearrange(recovered_features_flat, '(h w) b c -> b c h w', h=H, w=W)
        
        recovery_mask = should_recover.unsqueeze(1).float()
        recovered_features = recovered_features * recovery_mask
        
        similarity_norm = (text_similarity + 1.0) / 2.0
        recovery_weight_map = 0.7 + 0.3 * similarity_norm
        recovery_weight_map = recovery_weight_map * should_recover.float()
        recovery_weight_map = recovery_weight_map.unsqueeze(1)
        
        recovery_contribution = recovery_weight_map * recovered_features
        has_nan_inf_recovery = torch.isnan(recovery_contribution).any() or torch.isinf(recovery_contribution).any()
        if has_nan_inf_recovery:
            nan_inf_mask_recovery = torch.isnan(recovery_contribution) | torch.isinf(recovery_contribution)
            recovery_contribution = torch.where(nan_inf_mask_recovery,
                                               torch.zeros_like(recovery_contribution),
                                               recovery_contribution)
        
        final_features = features + recovery_contribution
        
        has_nan_inf_final = torch.isnan(final_features).any() or torch.isinf(final_features).any()
        if has_nan_inf_final:
            final_features = torch.clamp(final_features, min=-50.0, max=50.0)
            nan_inf_mask_final = torch.isnan(final_features) | torch.isinf(final_features)
            if nan_inf_mask_final.any():
                final_features = torch.where(nan_inf_mask_final,
                                            torch.zeros_like(final_features),
                                            final_features)
        
        if self.training:
            recovered_count = should_recover.sum().item()
            total_pruned = pruning_mask.sum().item()
            recovery_ratio = recovered_count / total_pruned if total_pruned > 0 else 0.0
            
            self._recovery_stats = {
                'recovered_count': recovered_count,
                'recovery_ratio': recovery_ratio,
                'avg_similarity': text_similarity[should_recover].mean().item() if should_recover.any() else 0.0,
                'avg_weight': recovery_weight_map[should_recover.unsqueeze(1)].mean().item() if should_recover.any() else 0.0,
                'total_pruned': total_pruned,
                'dvr_disabled': False
            }
        
        return final_features


class FeatureEnhancement(nn.Module):
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=False,
            dropout=dropout,
            bias=True
        )
        
        self.enhancement_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model)
        )
        
        self.text_proj = nn.Linear(d_model, d_model)
        self.vision_proj = nn.Linear(d_model, d_model)
        
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.vision_proj.weight)
        if self.text_proj.bias is not None:
            nn.init.constant_(self.text_proj.bias, 0.0)
        if self.vision_proj.bias is not None:
            nn.init.constant_(self.vision_proj.bias, 0.0)
        
        for m in self.enhancement_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self,
                features: torch.Tensor,
                text_features: torch.Tensor,
                pruning_mask: torch.Tensor = None,
                h: int = None,
                w: int = None) -> torch.Tensor:
        B_T, C, H, W = features.shape
        L = text_features.shape[0]
        
        features_flat = rearrange(features, 'b c h w -> (h w) b c')
        
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(features_flat)
        
        
        text_proj_expanded = text_proj
        
        enhanced_flat, _ = self.cross_attn(
            query=vision_proj,
            key=text_proj_expanded,
            value=text_proj_expanded,
            need_weights=False
        )
        
        text_vision_sim = (features_flat * enhanced_flat).sum(dim=-1, keepdim=True) / (features_flat.norm(dim=-1, keepdim=True) * enhanced_flat.norm(dim=-1, keepdim=True) + 1e-6)
        gate = 0.5 + 0.5 * torch.sigmoid(text_vision_sim * 5)
        fused_features = features_flat + gate * enhanced_flat
        
        enhanced_features_flat = self.enhancement_net(fused_features)
        
        if self.use_residual:
            enhancement_strength = (enhanced_features_flat.abs().sum(dim=-1, keepdim=True) / (features_flat.abs().sum(dim=-1, keepdim=True) + 1e-6))
            adaptive_residual_weight = 1.0 + 0.6 * torch.sigmoid(enhancement_strength - 1.0)
            enhanced_features_flat = features_flat + adaptive_residual_weight * enhanced_features_flat
        
        if pruning_mask is not None:
            features_nonzero = (features.abs().sum(dim=1) > 0)
            keep_mask = features_nonzero.float()
            keep_mask_flat = rearrange(keep_mask, 'b h w -> (h w) b')
            keep_mask_flat = keep_mask_flat.unsqueeze(-1)
            enhanced_features_flat = enhanced_features_flat * keep_mask_flat
        
        enhanced_features = rearrange(enhanced_features_flat, '(h w) b c -> b c h w', h=H, w=W)
        
        return enhanced_features

