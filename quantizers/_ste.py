import torch


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x
