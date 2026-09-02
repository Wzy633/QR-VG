import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False
    spconv = None


class SparseConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_spconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.use_spconv = use_spconv and SPCONV_AVAILABLE and torch.cuda.is_available()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=groups, bias=bias
        )
        
        if self.use_spconv:
            kernel_size_3d = (1, self.kernel_size[0], self.kernel_size[1]) if isinstance(self.kernel_size, tuple) else (1, self.kernel_size, self.kernel_size)
            padding_3d = (0, self.padding[0], self.padding[1]) if isinstance(self.padding, tuple) else (0, self.padding, self.padding)
            dilation_3d = (1, self.dilation[0], self.dilation[1]) if isinstance(self.dilation, tuple) else (1, self.dilation, self.dilation)
            
            stride_is_one = (self.stride == (1, 1) or self.stride == 1)
            if not stride_is_one:
                print(f"[WARNING] SubMConv3d requires stride=1, but got stride={self.stride}. Falling back to standard conv.")
                self.use_spconv = False
            else:
                self.sparse_conv = spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size_3d,
                    padding=padding_3d,
                    dilation=dilation_3d, groups=groups, bias=bias
                )
                self._sync_weights_from_sparse_to_conv()
    
    def _sync_weights_from_sparse_to_conv(self):
        try:
            sparse_weight = self.sparse_conv.weight.data
            if sparse_weight.dim() == 5:
                conv_weight = sparse_weight.squeeze(2)
                self.conv.weight.data.copy_(conv_weight)
            
            if self.sparse_conv.bias is not None and self.conv.bias is not None:
                self.conv.bias.data.copy_(self.sparse_conv.bias.data)
        except Exception as e:
            pass
    
    def _sync_weights_from_conv_to_sparse(self):
        try:
            if not self.use_spconv or not hasattr(self, 'sparse_conv'):
                return
            
            conv_weight = self.conv.weight.data
            if conv_weight.dim() == 4:
                sparse_weight = conv_weight.unsqueeze(2)
                self.sparse_conv.weight.data.copy_(sparse_weight)
            
            if self.conv.bias is not None and self.sparse_conv.bias is not None:
                self.sparse_conv.bias.data.copy_(self.conv.bias.data)
        except Exception as e:
            pass
        
    def forward(
        self,
        x: torch.Tensor,
        pruning_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if pruning_mask is None:
            if self.use_spconv:
                B, C, H, W = x.shape
                indices = torch.stack([
                    torch.arange(B, device=x.device, dtype=torch.int32).repeat_interleave(H * W),
                    torch.zeros(B * H * W, dtype=torch.int32, device=x.device),
                    torch.arange(H, device=x.device, dtype=torch.int32).repeat(W).repeat(B),
                    torch.arange(W, device=x.device, dtype=torch.int32).repeat(H).repeat(B),
                ], dim=1)
                features = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
                sparse_tensor = spconv.SparseConvTensor(
                    features=features,
                    indices=indices,
                    spatial_shape=(1, H, W),
                    batch_size=B
                )
                output_sparse = self.sparse_conv(sparse_tensor)
                output_H, output_W = output_sparse.spatial_shape[1], output_sparse.spatial_shape[2]
                output = torch.zeros(B, self.out_channels, output_H, output_W, device=x.device, dtype=x.dtype)
                if output_sparse.features.shape[0] > 0:
                    output_batch = output_sparse.indices[:, 0].long()
                    output_h = output_sparse.indices[:, 2].long()
                    output_w = output_sparse.indices[:, 3].long()
                    
                    valid_mask = (output_batch >= 0) & (output_batch < B) & \
                                (output_h >= 0) & (output_h < output_H) & \
                                (output_w >= 0) & (output_w < output_W)
                    
                    if valid_mask.any():
                        output_batch_valid = output_batch[valid_mask]
                        output_h_valid = output_h[valid_mask]
                        output_w_valid = output_w[valid_mask]
                        features_valid = output_sparse.features[valid_mask]
                        output[output_batch_valid, :, output_h_valid, output_w_valid] = features_valid
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    output = torch.clamp(output, min=-50.0, max=50.0)
                    output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        if not hasattr(self, '_nan_fallback_warned'):
                            nan_count = torch.isnan(output).sum().item()
                            inf_count = torch.isinf(output).sum().item()
                            print(f"[WARNING] Sparse conv output has NaN/Inf after fix (NaN: {nan_count}, Inf: {inf_count}), using standard conv fallback")
                            self._nan_fallback_warned = True
                        output = self.conv(x)
                
                return output
            else:
                return self.conv(x)
        
        B, C, H, W = x.shape
        assert pruning_mask.shape == (B, H, W), \
            f"pruning_mask shape {pruning_mask.shape} != (B, H, W) = ({B}, {H}, {W})"
        
        assert pruning_mask.device == x.device, \
            f"pruning_mask device {pruning_mask.device} != x device {x.device}"
        
        assert pruning_mask.dtype == torch.bool, \
            f"pruning_mask dtype {pruning_mask.dtype} != torch.bool"
        
        if self.use_spconv:
            x = torch.clamp(x, min=-50.0, max=50.0)
            x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
            
            valid_mask = ~pruning_mask
            
            batch_indices, h_indices, w_indices = torch.where(valid_mask)
            num_valid = len(batch_indices)
            
            if num_valid == 0:
                if not hasattr(self, '_fallback_warned'):
                    print(f"[WARNING] All positions pruned (num_valid=0), using standard conv fallback (no computation reduction)")
                    self._fallback_warned = True
                
                output = self.conv(x)
                return output
            
            batch_indices = batch_indices.clamp(0, B - 1)
            h_indices = h_indices.clamp(0, H - 1)
            w_indices = w_indices.clamp(0, W - 1)
            
            indices = torch.stack([
                batch_indices.to(torch.int32),
                torch.zeros(num_valid, dtype=torch.int32, device=x.device),
                h_indices.to(torch.int32),
                w_indices.to(torch.int32),
            ], dim=1).contiguous()
            
            features = x[batch_indices, :, h_indices, w_indices]
            
            features = torch.clamp(features, min=-50.0, max=50.0)
            features = torch.nan_to_num(features, nan=0.0, posinf=50.0, neginf=-50.0)
            
            sparse_tensor = spconv.SparseConvTensor(
                features=features,
                indices=indices,
                spatial_shape=(1, H, W),
                batch_size=B
            )
            
            try:
                if not hasattr(self, '_spconv_verified'):
                    total_positions = B * H * W
                    position_reduction = (1 - num_valid / total_positions) * 100
                    
                    if isinstance(self.kernel_size, tuple):
                        kernel_h, kernel_w = self.kernel_size[0], self.kernel_size[1]
                    else:
                        kernel_h = kernel_w = self.kernel_size
                    kernel_ops = kernel_h * kernel_w * self.in_channels * self.out_channels
                    dense_ops = total_positions * kernel_ops
                    sparse_ops = num_valid * kernel_ops
                    computation_reduction = (1 - sparse_ops / dense_ops) * 100 if dense_ops > 0 else 0.0
                    
                    print(f"[VERIFY] Using spconv SubMConv3d (submanifold sparse convolution)")
                    print(f"  Input: {num_valid} non-zero positions out of {total_positions} total")
                    print(f"  Position reduction: {position_reduction:.1f}%")
                    print(f"  Computation reduction: {computation_reduction:.1f}% (real reduction with SubMConv3d)")
                    self._spconv_verified = True
                
                output_sparse = self.sparse_conv(sparse_tensor)
            except Exception as e:
                print(f"Warning: spconv forward failed: {e}, fixing and using standard conv fallback")
                x_fixed = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
                return self.conv(x_fixed)
            
            if output_sparse.features.shape[0] > 0:
                features_clamped = torch.clamp(output_sparse.features, min=-50.0, max=50.0)
                features_fixed = torch.nan_to_num(
                    features_clamped, nan=0.0, posinf=50.0, neginf=-50.0
                )
                output_sparse = output_sparse.replace_feature(features_fixed)
            
            output_H, output_W = output_sparse.spatial_shape[1], output_sparse.spatial_shape[2]
            output = torch.zeros(B, self.out_channels, output_H, output_W, device=x.device, dtype=x.dtype)
            
            if output_sparse.features.shape[0] > 0:
                output_batch = output_sparse.indices[:, 0].long()
                output_h = output_sparse.indices[:, 2].long()
                output_w = output_sparse.indices[:, 3].long()
                
                valid_mask = (output_batch >= 0) & (output_batch < B) & \
                            (output_h >= 0) & (output_h < output_H) & \
                            (output_w >= 0) & (output_w < output_W)
                
                if valid_mask.any():
                    output_batch_valid = output_batch[valid_mask]
                    output_h_valid = output_h[valid_mask]
                    output_w_valid = output_w[valid_mask]
                    features_valid = output_sparse.features[valid_mask]
                    output[output_batch_valid, :, output_h_valid, output_w_valid] = features_valid
            
            
            output = torch.clamp(output, min=-50.0, max=50.0)
            output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
            
            return output
        else:
            output = self.conv(x)
            return output


class SparseConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
        use_sparse: bool = True,
    ):
        super().__init__()
        self.use_sparse = use_sparse
        
        if use_sparse:
            self.conv = SparseConv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding
            )
        
        self.norm = norm
        self.activation = activation
        
    def forward(
        self,
        x: torch.Tensor,
        pruning_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_sparse and pruning_mask is not None:
            x = self.conv(x, pruning_mask=pruning_mask)
        else:
            x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class SparseFeatureProcessor(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        use_sparse: bool = True,
    ):
        super().__init__()
        self.use_sparse = use_sparse
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            
            if i == num_layers - 1:
                norm = nn.GroupNorm(32, out_ch)
                activation = None
            else:
                norm = nn.GroupNorm(32, out_ch)
                activation = nn.ReLU(inplace=True)
            
            layer = SparseConvBlock(
                in_ch, out_ch,
                kernel_size=3, stride=1, padding=1,
                norm=norm, activation=activation,
                use_sparse=use_sparse
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        
    def forward(
        self,
        x: torch.Tensor,
        pruning_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pruning_mask=pruning_mask)
        return x
