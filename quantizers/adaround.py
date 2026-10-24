import torch
from torch import nn
from quantizers.uniform import UniformQuantizer


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive rounding (AdaRound, https://arxiv.org/abs/2004.10568) of a weight quantizer.

    :param uq: UniformQuantizer that provides the quantization parameters
    :param weight_tensor: weight used to initialize alpha

    round_mode is 'learned_hard_sigmoid' during reconstruction and 'nearest' afterwards.
    """

    def __init__(self, uq: UniformQuantizer, weight_tensor: torch.Tensor):
        super().__init__()
        self.n_levels = uq.n_levels
        self.scale = nn.Parameter(uq.scale)
        self.zero_point = nn.Parameter(uq.zero_point)

        self.round_mode = 'learned_hard_sigmoid'
        self.alpha = None
        self.soft_targets = False

        # stretch of the rectified sigmoid
        self.gamma, self.zeta = -0.1, 1.1
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.scale)
        else:
            x_floor = torch.floor(x / self.scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        x_quant = torch.clamp(x_int + self.zero_point, 0, 2 * self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.scale
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.scale)
        rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        self.alpha = nn.Parameter(alpha)

    def get_hard_value(self, x):
        init_shape = x.shape
        return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)
