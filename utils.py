import contextlib

import numpy as np
import PIL.Image
import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0")


def get_local_torch_device() -> torch.device:
  return DEVICE


def get_available_gpu_memory() -> float:
  free, _ = torch.cuda.mem_get_info(0)
  return free / (1 << 30)


class skip_init_modules:
  def __enter__(self):
    self._orig = {}
    for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
      self._orig[cls] = cls.reset_parameters
      cls.reset_parameters = lambda self: None

  def __exit__(self, *_):
    for cls, orig in self._orig.items():
      cls.reset_parameters = orig


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
  old = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(old)


def pil_to_numpy(image: PIL.Image.Image) -> np.ndarray:
  return np.array(image).astype(np.float32)[None] / 255.0


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
  if images.ndim == 3:
    images = images[..., None]
  return torch.from_numpy(images.transpose(0, 3, 1, 2))


def normalize(images: torch.Tensor) -> torch.Tensor:
  return 2.0 * images - 1.0
