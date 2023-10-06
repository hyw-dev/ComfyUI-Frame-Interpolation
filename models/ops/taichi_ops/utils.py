import platform
import torch
def to_shared_memory(tensors: tuple[torch.Tensor]):
    return [tensor.cpu() for tensor in tensors if tensor is not None]

def to_device(tensors: tuple[torch.Tensor], device: torch.device):
    return [tensor.to(device) for tensor in tensors if tensor is not None]