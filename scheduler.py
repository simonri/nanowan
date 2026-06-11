"""Flow-matching Euler discrete scheduler (minimal, no diffusers dependency)."""

import numpy as np
import torch


class FlowMatchEulerDiscreteScheduler:
  def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
    self.num_train_timesteps = num_train_timesteps
    self.shift = shift

    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    self.sigmas = sigmas.to("cpu")
    self.timesteps = sigmas * num_train_timesteps
    self.sigma_min = self.sigmas[-1].item()
    self.sigma_max = self.sigmas[0].item()
    self._step_index: int | None = None

  @property
  def step_index(self) -> int | None:
    return self._step_index

  def set_timesteps(self, num_inference_steps: int, device: str | torch.device = "cpu") -> None:
    timesteps_array = np.linspace(
      self.sigma_max * self.num_train_timesteps,
      self.sigma_min * self.num_train_timesteps,
      num_inference_steps,
    )
    sigmas_array = timesteps_array / self.num_train_timesteps
    sigmas_array = self.shift * sigmas_array / (1 + (self.shift - 1) * sigmas_array)

    sigmas_tensor = torch.from_numpy(sigmas_array.astype(np.float32)).to(device=device)
    sigmas_tensor = torch.cat([sigmas_tensor, torch.zeros(1, device=device)])

    self.timesteps = sigmas_tensor[:-1] * self.num_train_timesteps
    self.sigmas = sigmas_tensor
    self._step_index = None

  def index_for_timestep(self, timestep: float | torch.FloatTensor) -> int:
    indices = (self.timesteps == timestep).nonzero()
    pos = 1 if len(indices) > 1 else 0
    return indices[pos].item()

  def _init_step_index(self, timestep) -> None:
    if isinstance(timestep, torch.Tensor):
      timestep = timestep.to(self.timesteps.device)
    self._step_index = self.index_for_timestep(timestep)

  def step(
    self,
    model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor,
  ) -> torch.FloatTensor:
    if self._step_index is None:
      self._init_step_index(timestep)

    sample = sample.to(torch.float32)
    sigma = self.sigmas[self._step_index]
    sigma_next = self.sigmas[self._step_index + 1]
    dt = sigma_next - sigma
    prev_sample = sample + dt * model_output
    self._step_index += 1
    return prev_sample.to(model_output.dtype)
