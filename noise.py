import abc
import torch
import torch.nn as nn

class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  
  def forward(self, t):
    return self.compute_loss_scaling_and_move_chance(t)

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def compute_loss_scaling_and_move_chance(self, t):
    loss_scaling = - 1 / t
    return loss_scaling, t
