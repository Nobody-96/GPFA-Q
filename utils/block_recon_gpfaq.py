"""GPFA-Q block reconstruction.

For every transformer block the reconstruction objective is

    L = L_GPR / L_GPR(0) + L_Diag / L_Diag(0) + lambda(t) * L_SGR / L_SGR(0)

    L_GPR  = || G_alpha |dZ|^T ||_F^2 / (alpha * B)   (G_alpha: alpha sampled |dKL/dZ| rows)
    L_Diag = mean( |dZ|^2 * E[g^2] )
    L_SGR  = L_GPR evaluated on the hard-rounded output (straight-through to the soft one)

where dZ is the block output error. The gradient basis is collected once per block, at the first
iteration. L_GPR and L_Diag are divided by their first-iteration values, L_SGR by its value at the
first SGR iteration. lambda(t) grows linearly from 0 to lambda_max after a warm-up. The AdaRound
rounding regulariser is added.

Ablations (--ablation in test_quant.py):
    gpfaq     GPR + Diag + SGR
    gpr       GPR + Diag
    baseline  normalised MSE (no gradient statistics are collected)
Structural blocks (patch_embed, head) always use the normalised MSE.
"""

import logging
from types import MethodType

import torch
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import window_partition, window_reverse

from utils.calibrator import QuantCalibrator
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul

ITERS = 20000  # reconstruction iterations per block


def patch_embed_forward(self, x):
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    else:
        x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    return x


def vit_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


def swin_block_forward(self, x):
    B, H, W, C = x.shape
    shortcut = x
    x = self.norm1(x)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = shortcut + self.drop_path(x)
    x = x.reshape(B, -1, C)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    x = x.reshape(B, H, W, C)
    return x


def swin_patchmerging_forward(self, x):
    B, H, W, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
    x = self.norm(x)
    x = self.reduction(x)
    return x


# Reconstruction units, in model order: these modules and every module named 'head'.
BLOCK_TYPES = (
    timm.layers.patch_embed.PatchEmbed,
    timm.models.vision_transformer.Block,
    timm.models.swin_transformer.SwinTransformerBlock,
    timm.models.swin_transformer.PatchMerging,
)


def prepare_block(module):
    module.raw_input = module.tmp_input = None
    module.raw_out = module.tmp_out = None
    module.raw_grad = module.tmp_grad = None
    module.quanted_input = None
    if isinstance(module, timm.layers.patch_embed.PatchEmbed):
        module.forward = MethodType(patch_embed_forward, module)
    elif isinstance(module, timm.models.vision_transformer.Block):
        module.forward = MethodType(vit_block_forward, module)
    elif isinstance(module, timm.models.swin_transformer.SwinTransformerBlock):
        module.forward = MethodType(swin_block_forward, module)
    elif isinstance(module, timm.models.swin_transformer.PatchMerging):
        module.forward = MethodType(swin_patchmerging_forward, module)


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class GPFAQLoss:
    def __init__(self, wqs, max_count, use_gpr, use_sgr, lambda_max, sgr_warmup,
                 weight=0.01, b_range=(20, 2), warmup=0.2, norm_floor=1e-8):
        self.wqs = wqs
        self.max_count = max_count
        self.use_gpr = use_gpr
        self.use_sgr = use_sgr
        self.lambda_max = lambda_max
        self.weight = weight
        self.norm_floor = norm_floor
        self.loss_start = max_count * warmup
        self.sgr_start = max_count * sgr_warmup
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.init_loss_1 = None
        self.init_loss_2 = None
        self.init_loss_sgr = None

    def __call__(self, pred, tgt, grad=None, grad2=None, pred_hard=None):
        """pred/tgt: block outputs; grad: gradient basis (alpha, D); grad2: E[g^2] (1, D);
        pred_hard: output with hard rounding (only needed for SGR)."""
        self.count += 1
        B = pred.shape[0]

        if self.use_gpr:
            cha = (pred - tgt).abs().reshape(B, -1)
            loss_1 = (grad @ cha.T).pow(2).sum() / (grad.shape[0] * B)
            loss_2 = (cha.pow(2) * grad2).mean()
            if self.count == 1:
                self.init_loss_1 = loss_1.detach().clamp(min=self.norm_floor)
                self.init_loss_2 = loss_2.detach().clamp(min=self.norm_floor)
            rec_loss = loss_1 / self.init_loss_1 + loss_2 / self.init_loss_2
        else:
            loss_1 = F.mse_loss(pred, tgt)
            if self.count == 1:
                self.init_loss_1 = loss_1.detach().clamp(min=self.norm_floor)
            rec_loss = loss_1 / self.init_loss_1

        # SGR: forward value of the hard-rounded output, gradient of the soft one.
        lambda_cur = 0.0
        if self.use_sgr and self.count > self.sgr_start:
            pred_ste = pred + (pred_hard - pred).detach()
            cha_ste = (pred_ste - tgt).abs().reshape(B, -1)
            loss_sgr = (grad @ cha_ste.T).pow(2).sum() / (grad.shape[0] * B)
            if self.init_loss_sgr is None:
                self.init_loss_sgr = loss_sgr.detach().clamp(min=self.norm_floor)
            loss_sgr_norm = loss_sgr / self.init_loss_sgr
            progress = (self.count - self.sgr_start) / (self.max_count - self.sgr_start)
            lambda_cur = self.lambda_max * progress
            rec_loss = rec_loss + lambda_cur * loss_sgr_norm

        # AdaRound rounding regulariser
        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            b = round_loss = 0
        else:
            round_loss = 0
            for wq in self.wqs:
                rv = wq.get_soft_targets()
                round_loss += self.weight * (1 - ((rv - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        if self.count == 1 or self.count % 500 == 0:
            print(f'Total loss:\t{float(total_loss):.3f} (rec:{float(rec_loss):.3f}, '
                  f'lambda:{lambda_cur:.3f}, round:{float(round_loss):.3f})\tb={b:.2f}\tcount={self.count}')
        return total_loss


class GPFAQBlockReconstructor(QuantCalibrator):
    def __init__(self, model, calib_loader, batch_size, temp, alpha, lambda_max, sgr_warmup,
                 drop_prob, use_gpr, use_sgr):
        assert use_gpr or not use_sgr, 'SGR uses the GPR gradient basis'
        super().__init__(model, calib_loader)
        self.batch_size = batch_size
        self.temperature = temp
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.sgr_warmup = sgr_warmup
        self.iters = ITERS
        self.drop_prob = drop_prob
        self.use_gpr = use_gpr
        self.use_sgr = use_sgr
        self.blocks = {}
        self.quanted_blocks = []
        self.raw_pred_softmaxs = None
        for name, module in self.model.named_modules():
            if isinstance(module, BLOCK_TYPES) or name.split('.')[-1] == 'head':
                self.blocks[name] = module
                prepare_block(module)

    def set_block_mode(self, block, mode='raw'):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    def wrap_quantizers_in_net(self, block, name):
        print('wrapping quantizers in {} ...'.format(name))
        for _, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer,
                                                           weight_tensor = module.weight.view(module.n_V, module.crb_rows, module.in_features))
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer,
                                                           weight_tensor = module.weight.view(module.weight.shape[0], -1))
                module.w_quantizer.soft_targets = True

    def set_qdrop(self, block, prob):
        for _, module in block.named_modules():
            if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                module.a_quantizer.drop_prob = prob
            elif isinstance(module, MinMaxQuantMatMul):
                module.A_quantizer.drop_prob = prob
                module.B_quantizer.drop_prob = prob

    def init_block_raw_data(self, block, name, device):
        self.init_block_raw_inp_outp(block, device)
        if 'patch_embed' in name:
            block.quanted_input = block.raw_input
        else:
            self.init_block_quanted_input(block, device)
        block.raw_input, block.raw_out = block.raw_input.to(device), block.raw_out.to(device)
        block.quanted_input = block.quanted_input.to(device)

    def init_block_raw_inp_outp(self, block, device):
        logging.info('initializing raw input and raw output ...')
        for _block in self.blocks.values():
            self.set_block_mode(_block, 'raw')
        hooks = []
        hooks.append(block.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.register_forward_hook(self.single_input_forward_hook))
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.model(inp)
                torch.cuda.empty_cache()
        block.raw_out = torch.cat(block.tmp_out, dim=0)
        block.raw_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input, block.tmp_out = None, None
        for hook in hooks:
            hook.remove()
        torch.cuda.empty_cache()

    def init_block_quanted_input(self, block, device):
        logging.info('initializing quanted input ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        hook = block.register_forward_hook(self.single_input_forward_hook)
        with torch.no_grad():
            for i, (inp, target) in enumerate(self.calib_loader):
                inp = inp.to(device)
                pred = self.model(inp)
        torch.cuda.empty_cache()
        block.quanted_input = torch.cat(block.tmp_input, dim=0)
        block.tmp_input = None
        hook.remove()
        for _block in self.blocks.values():
            self.set_block_mode(_block, 'raw')

    def collect_gradient_basis(self, block, device):
        """Store alpha sampled |dKL/dZ| rows in block.raw_grad and E[g^2] in block.raw_grad2."""
        if self.raw_pred_softmaxs is None:
            # Target distribution of the KL loss. It is computed once, at the first call,
            # i.e. while the first reconstructed transformer block is already quantized.
            self.raw_pred_softmaxs = []
            with torch.no_grad():
                for inp, _ in self.calib_loader:
                    pred = self.model(inp.to(device)) / self.temperature
                    self.raw_pred_softmaxs.append(F.softmax(pred, dim=-1).detach().cpu())

        print('collecting gradient basis ...')
        hook = block.register_full_backward_hook(self.grad_hook)
        for i, (inp, _) in enumerate(self.calib_loader):
            self.model.zero_grad()
            pred = self.model(inp.to(device)) / self.temperature
            loss = F.kl_div(F.log_softmax(pred, dim=-1),
                            self.raw_pred_softmaxs[i].to(device),
                            reduction='batchmean')
            loss.backward()
            torch.cuda.empty_cache()
        hook.remove()

        raw_grad_all = torch.cat(block.tmp_grad, dim=0)
        raw_grad_all = raw_grad_all.reshape(raw_grad_all.shape[0], -1).abs()
        block.tmp_grad = None
        idx = torch.randperm(raw_grad_all.shape[0])[:self.alpha]
        block.raw_grad = raw_grad_all[idx]
        block.raw_grad2 = raw_grad_all.pow(2).mean(0, keepdim=True)
        del raw_grad_all
        torch.cuda.empty_cache()

    def reconstruct_single_block(self, name, block, device):
        self.wrap_quantizers_in_net(block, name)
        self.set_block_mode(block, 'quant_forward')
        for _, module in block.named_modules():
            if hasattr(module, 'training_mode'):
                module.init_training()
        self.set_qdrop(block, self.drop_prob)

        w_params, a_params = [], []
        for _, module in block.named_modules():
            if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d)):
                w_params += [module.w_quantizer.alpha]
                module.a_quantizer.scale.requires_grad = True
                a_params += [module.a_quantizer.scale]
            elif isinstance(module, MinMaxQuantMatMul):
                for q in [module.A_quantizer, module.B_quantizer]:
                    q.scale.requires_grad = True
                    a_params += [q.scale]
        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=4e-5)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=self.iters, eta_min=0.)

        # patch_embed and head are reconstructed with the normalised MSE only.
        is_structural = 'head' in name or 'patch_embed' in name
        use_gpr = self.use_gpr and not is_structural
        use_sgr = self.use_sgr and not is_structural
        wqs = [m.w_quantizer for _, m in block.named_modules() if hasattr(m, 'w_quantizer')]
        loss_func = GPFAQLoss(wqs, max_count=self.iters, use_gpr=use_gpr, use_sgr=use_sgr,
                              lambda_max=self.lambda_max, sgr_warmup=self.sgr_warmup)
        cur_grad = cur_grad2 = None

        for it in range(self.iters):
            idx = torch.randperm(block.raw_input.size(0))[:self.batch_size]
            idx = idx.pin_memory().to(device, non_blocking=True)

            # QDrop: each element of the block input is quantized with probability drop_prob
            q = block.quanted_input[idx]
            fp = block.raw_input[idx]
            cur_inp = torch.where(torch.rand_like(q) < self.drop_prob, q, fp)
            cur_out = block.raw_out[idx]

            if use_gpr and it == 0:
                self.collect_gradient_basis(block, device)
                cur_grad = block.raw_grad.to(device)
                cur_grad2 = block.raw_grad2.to(device)

            out_hard = None
            if use_sgr and it >= self.iters * self.sgr_warmup:
                with torch.no_grad():
                    for wq in wqs:
                        wq.soft_targets = False
                    out_hard = block(cur_inp)
                    for wq in wqs:
                        wq.soft_targets = True

            w_optimizer.zero_grad()
            a_optimizer.zero_grad()
            out_quant = block(cur_inp)
            err = loss_func(out_quant, cur_out, cur_grad, cur_grad2, pred_hard=out_hard)
            err.backward()
            w_optimizer.step()
            a_optimizer.step()
            a_scheduler.step()

        torch.cuda.empty_cache()
        # Finish optimization, use hard rounding.
        for _, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input
        if hasattr(block, 'raw_grad2'):
            del block.raw_grad2
        del cur_grad, cur_grad2
        torch.cuda.empty_cache()

    def reconstruct_model(self):
        device = next(self.model.parameters()).device
        for _, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'raw'

        for name, block in self.blocks.items():
            logging.info(f'reconstructing {name} ...')
            self.init_block_raw_data(block, name, device)
            self.reconstruct_single_block(name, block, device)
            self.quanted_blocks.append(name)
            logging.info(f'finished reconstructing {name}.')

        for _, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'quant_forward'
            if hasattr(module, 'w_quantizer'):
                module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
                del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"
