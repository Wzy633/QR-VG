
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, pruning_mask=None):
        """
        添加pruning_mask参数用于稀疏注意力
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :param pruning_mask                (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for pruned elements, False for non-pruned elements 
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        if pruning_mask is not None:
            value = value.masked_fill(pruning_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        if pruning_mask is not None and os.environ.get("PRVG_PY_SPARSE_MASKING", "0") == "1":
            N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
            spatial_shapes_list = input_spatial_shapes.tolist()
            level_start_list = input_level_start_index.tolist()
            for lvl in range(n_levels):
                H_, W_ = spatial_shapes_list[lvl]
                level_start = level_start_list[lvl]
                level_end = level_start + H_ * W_
                pruning_mask_lvl = pruning_mask[:, level_start:level_end]
                pruning_mask_spatial = pruning_mask_lvl.view(N, H_, W_)
                sampling_locs_lvl = sampling_locations[:, :, :, lvl, :, :]
                x_coords = (sampling_locs_lvl[:, :, :, :, 0] * (W_ - 1)).long().clamp(0, W_ - 1)
                y_coords = (sampling_locs_lvl[:, :, :, :, 1] * (H_ - 1)).long().clamp(0, H_ - 1)
                N_idx = torch.arange(N, device=pruning_mask_spatial.device).view(N, 1, 1, 1).expand(N, Len_q, n_heads, n_points)
                is_pruned = pruning_mask_spatial[N_idx, y_coords, x_coords]
                weight_start_idx = lvl * self.n_points
                weight_end_idx = (lvl + 1) * self.n_points
                attention_weights[:, :, :, weight_start_idx:weight_end_idx] = \
                    attention_weights[:, :, :, weight_start_idx:weight_end_idx].masked_fill(is_pruned, float('-inf'))
            all_inf_mask = torch.isinf(attention_weights) & (attention_weights < 0)
            all_inf_per_query_head = all_inf_mask.all(dim=-1, keepdim=True)
            attention_weights = torch.where(
                all_inf_per_query_head.expand_as(attention_weights),
                torch.full_like(attention_weights, -1e10),
                attention_weights
            )

        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        if pruning_mask is not None:
            if not hasattr(self, '_cuda_sparse_impl_printed'):
                if os.environ.get("PRVG_VERIFY", "0") == "1":
                    print("[VERIFY] Using CUDA implementation with sparse attention - truly skips pruned positions before sampling")
                self._cuda_sparse_impl_printed = True
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, 
                sampling_locations, attention_weights, self.im2col_step, pruning_mask
            )
        else:
            if not hasattr(self, '_cuda_impl_printed'):
                if os.environ.get("PRVG_VERIFY", "0") == "1":
                    print("[VERIFY] Using CUDA implementation (MSDeformAttnFunction) - standard implementation")
                self._cuda_impl_printed = True
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, 
                sampling_locations, attention_weights, self.im2col_step, None
            )
        
        output = self.output_proj(output)

        return output, sampling_locations, attention_weights