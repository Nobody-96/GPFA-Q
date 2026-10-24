from types import MethodType
from torch import nn
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from quant_layers import (AsymmetricallyBatchingQuantConv2d, AsymmetricallyBatchingQuantLinear,
                          AsymmetricallyChannelWiseBatchingQuantLinear, AsymmetricallyBatchingQuantMatMul)

# scale search of the calibration stage
SEARCH_ROUND = 3
EQ_N = 128


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def vit_attn_forward(self, x):
    B, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q, k = self.q_norm(q), self.k_norm(k)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def swin_attn_forward(self, x, mask=None):
    B_, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def wrap_modules_in_net(model, cfg, reparam=False):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attn_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(swin_attn_forward, module)

    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        father_module = module_dict[father_name]

        if isinstance(module, nn.Conv2d):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantConv2d(
                in_channels = module.in_channels, 
                out_channels = module.out_channels,
                kernel_size = module.kernel_size,
                stride = module.stride,
                mode = 'raw',
                w_bit = cfg.w_bit,
                calib_batch_size = cfg.calib_batch_size,
                eq_n = EQ_N,
            )
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, MatMul):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantMatMul(
                A_bit = cfg.a_bit,
                B_bit = cfg.a_bit,
                mode = 'raw',
                calib_batch_size = cfg.calib_batch_size,
                search_round = SEARCH_ROUND,
                eq_n = EQ_N,
                num_heads = father_module.num_heads,
            )
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, nn.Linear):
            linear_kwargs = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'mode': 'raw',
                'w_bit': cfg.w_bit,
                'a_bit': cfg.a_bit,
                'calib_batch_size': cfg.calib_batch_size,
                'search_round': SEARCH_ROUND,
                'eq_n': EQ_N,
                'n_V': 3 if 'qkv' in name else 1,
            }
            idx = idx + 1 if idx != 0 else idx
            if reparam and ('qkv' in name or 'reduction' in name or 'fc1' in name):
                idxx = father_name.rfind('.')
                idxx = 0 if idxx == -1 else idxx
                grandfather_module = module_dict[father_name[:idxx]]
                new_module = AsymmetricallyChannelWiseBatchingQuantLinear(
                    **linear_kwargs, 
                )
                if 'qkv' in name:
                    new_module.prev_layer = grandfather_module.norm1
                if 'fc1' in name:
                    new_module.prev_layer = grandfather_module.norm2
                if 'reduction' in name:
                    new_module.prev_layer = father_module.norm
            else: 
                new_module = AsymmetricallyBatchingQuantLinear(
                    **linear_kwargs,
                )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            setattr(father_module, name[idx:], new_module)
    return model


def wrap_reparamed_modules_in_net(model):
    module_dict = {}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        father_module = module_dict[father_name]

        if isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            idx = idx + 1 if idx != 0 else idx
            linear_kwargs = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'mode': module.mode,
                'w_bit': module.w_quantizer.n_bits,
                'a_bit': module.a_quantizer.n_bits,
                'calib_batch_size': module.calib_batch_size,
                'search_round': module.search_round,
                'eq_n': module.eq_n,
                'n_V': module.n_V,
            }
            new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)
            if (new_module.a_quantizer.scale.shape != module.a_quantizer.scale.shape):
                new_module.a_quantizer.scale.data = module.a_quantizer.scale.data.clone()
            new_module.load_state_dict(module.state_dict())
            new_module.calibrated = True
            new_module.a_quantizer.inited = True
            new_module.w_quantizer.inited = True
            setattr(father_module, name[idx:], new_module)
    return model
    