import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers.uniform import UniformQuantizer


class MinMaxQuantConv2d(nn.Conv2d):
    """
    Conv2d with quantized weights. It is only used for the patch embedding, whose input (the image)
    is not quantized; a_quantizer only holds the input scale that is stored in checkpoints.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mode = mode
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, channel_wise = False)
        self.a_quantizer = UniformQuantizer(n_bits = 8, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False

    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        else:
            raise NotImplementedError
        return out

    def quant_forward(self,x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.conv2d(x, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out


class PTQSLQuantConv2d(MinMaxQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, padding_mode, mode, w_bit)
        self.eq_n = eq_n
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))

    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2

    def quant_weight_bias(self):
        oc, ic, kw, kh = self.weight.data.shape
        w_sim = self.w_quantizer(self.weight.view(oc, ic * kw * kh)).view(oc, ic, kw, kh)
        return w_sim, self.bias


class PTQSLBatchingQuantConv2d(PTQSLQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 calib_batch_size = 32,
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, padding_mode, mode, w_bit, eq_n)
        self.calib_batch_size = calib_batch_size

    def _initialize_calib_parameters(self):
        """
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input.shape[0]
        memory = torch.cuda.get_device_properties(0).total_memory // 2
        numel = (2 * self.raw_input[:self.calib_batch_size].numel() +
                 2 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))

    def _initialize_activation_scale(self):
        # Only stored in checkpoints: the input is not quantized.
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            a_scale_=(x_.abs().max() / (self.a_quantizer.n_levels - 0.5)).detach().view(1, 1)
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = torch.cat(tmp_a_scales, dim=1).amax(dim=1, keepdim=False).view(1, 1, 1, 1)
        self.a_quantizer.scale.data.copy_(tmp_a_scale) # shape: (1, 1, 1, 1)
        self.a_quantizer.inited = True


class AsymmetricallyBatchingQuantConv2d(PTQSLBatchingQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 calib_batch_size = 32,
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, padding_mode, mode, w_bit, calib_batch_size, eq_n)

        del self.w_quantizer
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, channel_wise = True)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros((self.out_channels, 1)))

    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed] # shape: (parallel_eq_n, oc, 1)
                cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
                # quantize weight and bias
                oc,ic,kw,kh = self.weight.data.shape
                w_sim = self.weight.view(oc, -1).unsqueeze(0) # shape: (1, oc, ic*kw*kh)
                w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clamp(0, 2 * self.w_quantizer.n_levels - 1)
                w_sim = (w_quant - cur_w_zero_point).mul_(cur_w_scale) # shape: (parallel_eq_n,oc,ic*kw*kh)
                w_sim = w_sim.view(-1,ic,kw,kh) # shape: parallel_eq_n*oc,ic,kw,kh
                bias_sim = self.bias.repeat(p_ed-p_st)
                # calculate similarity and store them
                out_sim = F.conv2d(x, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B,parallel_eq_n*oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, [3,4]) # shape: B,parallel_eq_n,oc
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: (1,parallel_eq_n,oc)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: (1,eq_n,oc)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: (eq_n,oc)
        best_index = batch_similarities.argmax(dim=0).reshape(1, -1, 1) # shape: (1,oc,1)
        tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
        tmp_w_zero_point = torch.gather(weight_zero_point_candidates, dim=0, index=best_index)
        self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
        self.w_quantizer.zero_point.data.copy_(tmp_w_zero_point.squeeze(dim=0))
        return best_index

    def calculate_percentile_weight_candidates(self, l=0.99, r=0.9999, k=0.05):
        pct = torch.tensor([l + (r - l) * (i / (self.eq_n - 1))**k for i in range(self.eq_n)] + [1.0])
        w_uppers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: eq_n, out_channels, 1
        w_lowers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), (1-pct).to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: eq_n, out_channels, 1
        return w_uppers_candidates, w_lowers_candidates

    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        self._initialize_activation_scale()

        w_uppers_candidates, w_lowers_candidates = self.calculate_percentile_weight_candidates(l=0.99, r=0.9999, k=0.5)
        weight_scale_candidates = ((w_uppers_candidates - w_lowers_candidates) / (2 * self.w_quantizer.n_levels - 1)).contiguous().cuda()
        weight_zero_point_candidates = -(w_lowers_candidates / weight_scale_candidates).round().contiguous().cuda()
        w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
        self.w_quantizer.inited = True

        # search the lower bound with the best upper bound fixed, then the upper bound
        for ee in range(2):
            if ee % 2 == 0:
                w_uppers_candidates_ = torch.gather(w_uppers_candidates, dim=0, index=w_best_index)
                w_lowers_candidates_ = w_lowers_candidates
            else:
                w_uppers_candidates_ = w_uppers_candidates
                w_lowers_candidates_ = torch.gather(w_lowers_candidates, dim=0, index=w_best_index)
            weight_scale_candidates = ((w_uppers_candidates_ - w_lowers_candidates_) / (2 * self.w_quantizer.n_levels - 1)).contiguous().cuda()
            weight_zero_point_candidates = -(w_lowers_candidates_ / weight_scale_candidates).round().contiguous().cuda()
            w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out
