"""
spconv 性能分析工具
使用 CUDA events 精确测量稀疏卷积的执行时间
"""
import torch
import time
from collections import defaultdict
from typing import Dict, List, Optional

class SparseConvProfiler:
    """稀疏卷积性能分析器"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats = defaultdict(list)
        self.total_stats = {
            'sparse_conv_time': 0.0,
            'dense_conv_time': 0.0,
            'format_convert_time': 0.0,
            'sparse_conv_calls': 0,
            'dense_conv_calls': 0,
            'total_positions': 0,
            'valid_positions': 0,
        }
        
    def start_timer(self, device: torch.device):
        """开始计时（使用CUDA events）"""
        if not self.enabled or device.type != 'cuda':
            return None
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event
    
    def end_timer(self, start_event, device: torch.device):
        """结束计时并返回耗时（ms）"""
        if start_event is None or device.type != 'cuda':
            return 0.0
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    
    def record_sparse_conv(
        self,
        num_valid: int,
        total_positions: int,
        sparse_conv_time: float,
        format_convert_time: float = 0.0
    ):
        """记录稀疏卷积的统计信息"""
        if not self.enabled:
            return
        
        self.total_stats['sparse_conv_time'] += sparse_conv_time
        self.total_stats['format_convert_time'] += format_convert_time
        self.total_stats['sparse_conv_calls'] += 1
        self.total_stats['total_positions'] += total_positions
        self.total_stats['valid_positions'] += num_valid
        
        self.stats['sparse_conv'].append({
            'num_valid': num_valid,
            'total_positions': total_positions,
            'sparse_conv_time': sparse_conv_time,
            'format_convert_time': format_convert_time,
            'pruning_ratio': 1.0 - num_valid / total_positions if total_positions > 0 else 0.0
        })
    
    def record_dense_conv(self, dense_conv_time: float):
        """记录标准卷积的统计信息"""
        if not self.enabled:
            return
        
        self.total_stats['dense_conv_time'] += dense_conv_time
        self.total_stats['dense_conv_calls'] += 1
        
        self.stats['dense_conv'].append({
            'dense_conv_time': dense_conv_time
        })
    
    def get_summary(self) -> Dict:
        """获取性能统计摘要"""
        if not self.enabled or self.total_stats['sparse_conv_calls'] == 0:
            return {}
        
        total_calls = self.total_stats['sparse_conv_calls'] + self.total_stats['dense_conv_calls']
        avg_sparse_time = self.total_stats['sparse_conv_time'] / self.total_stats['sparse_conv_calls'] if self.total_stats['sparse_conv_calls'] > 0 else 0.0
        avg_dense_time = self.total_stats['dense_conv_time'] / self.total_stats['dense_conv_calls'] if self.total_stats['dense_conv_calls'] > 0 else 0.0
        
        total_positions = self.total_stats['total_positions']
        valid_positions = self.total_stats['valid_positions']
        avg_pruning_ratio = 1.0 - valid_positions / total_positions if total_positions > 0 else 0.0
        
        theoretical_speedup = 1.0 / (1.0 - avg_pruning_ratio) if avg_pruning_ratio < 1.0 else 1.0
        
        actual_speedup = avg_dense_time / avg_sparse_time if avg_sparse_time > 0 else 0.0
        
        return {
            'total_calls': total_calls,
            'sparse_conv_calls': self.total_stats['sparse_conv_calls'],
            'dense_conv_calls': self.total_stats['dense_conv_calls'],
            'avg_pruning_ratio': avg_pruning_ratio,
            'total_sparse_time_ms': self.total_stats['sparse_conv_time'],
            'total_dense_time_ms': self.total_stats['dense_conv_time'],
            'total_format_convert_time_ms': self.total_stats['format_convert_time'],
            'avg_sparse_time_ms': avg_sparse_time,
            'avg_dense_time_ms': avg_dense_time,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'speedup_efficiency': actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0.0,
        }
    
    def print_summary(self):
        """打印性能统计摘要"""
        summary = self.get_summary()
        if not summary:
            print("⚠️  没有性能统计数据")
            return
        
        print("\n" + "="*80)
        print("spconv 性能分析报告")
        print("="*80)
        print(f"总调用次数: {summary['total_calls']}")
        print(f"  稀疏卷积: {summary['sparse_conv_calls']} 次")
        print(f"  标准卷积: {summary['dense_conv_calls']} 次")
        print(f"\n平均剪枝比例: {summary['avg_pruning_ratio']:.2%}")
        print(f"\n总执行时间:")
        print(f"  稀疏卷积: {summary['total_sparse_time_ms']:.2f} ms")
        print(f"  标准卷积: {summary['total_dense_time_ms']:.2f} ms")
        print(f"  格式转换: {summary['total_format_convert_time_ms']:.2f} ms")
        print(f"\n平均执行时间:")
        print(f"  稀疏卷积: {summary['avg_sparse_time_ms']:.3f} ms/次")
        print(f"  标准卷积: {summary['avg_dense_time_ms']:.3f} ms/次")
        print(f"\n加速比:")
        print(f"  理论加速比: {summary['theoretical_speedup']:.2f}x")
        print(f"  实际加速比: {summary['actual_speedup']:.2f}x")
        print(f"  加速效率: {summary['speedup_efficiency']:.2%}")
        print("="*80 + "\n")
    
    def reset(self):
        """重置统计信息"""
        self.stats.clear()
        for key in self.total_stats:
            self.total_stats[key] = 0.0 if isinstance(self.total_stats[key], float) else 0

_global_profiler = None

def get_profiler(enabled: bool = True) -> SparseConvProfiler:
    """获取全局性能分析器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = SparseConvProfiler(enabled=enabled)
    return _global_profiler

def reset_profiler():
    """重置全局性能分析器"""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()
