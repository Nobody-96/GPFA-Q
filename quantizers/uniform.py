import torch
import torch.nn as nn
from quantizers._ste import round_ste


class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.inited = False
        self.training_mode = False

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False

    def forward(self, x):
        assert self.inited
        if self.training_mode:
            x_orig = x
        x_int = round_ste(x / self.scale) if self.training_mode else torch.round(x / self.scale)
        x_quant = (x_int + round_ste(self.zero_point)).clamp(0, 2 * self.n_levels - 1)
        x_dequant = (x_quant - round_ste(self.zero_point)) * self.scale
        if self.training_mode:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant
