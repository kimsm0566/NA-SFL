"""CUDA device selection for NA-SFL; CPU training fallback is disabled."""
import torch


def require_cuda_device(index: int) -> torch.device:
    if not isinstance(index, int) or index < 0:
        raise RuntimeError("CUDA device index must be a nonnegative integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} is unavailable; visible count is {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)
