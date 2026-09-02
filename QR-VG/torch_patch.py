import torch
import numpy as np

def frombuffer(buffer, dtype, count=-1, offset=0):
    if isinstance(buffer, (bytes, bytearray)):
        buffer = np.frombuffer(buffer, dtype=np.uint8)
    
    if count > 0:
        buffer = buffer[offset:offset+count]
    else:
        buffer = buffer[offset:]
    
    return torch.from_numpy(buffer)

class DummyParametrizations:
    @staticmethod
    def weight_norm(*args, **kwargs):
        def dummy_weight_norm(module, name='weight', dim=0):
            return module
        return dummy_weight_norm

if not hasattr(torch, 'frombuffer'):
    torch.frombuffer = frombuffer
    print("已为 PyTorch 1.8.1 添加 torch.frombuffer 方法")

if not hasattr(torch.nn.utils, 'parametrizations'):
    torch.nn.utils.parametrizations = DummyParametrizations()
    print("已为 PyTorch 1.8.1 添加 nn.utils.parametrizations 模块")
