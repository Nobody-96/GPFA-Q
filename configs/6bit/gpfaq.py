# GPFA-Q config: W6/A6. Used for the 6-bit setting of the paper's Fig. 2(a) (Baseline, --ablation baseline).
#
#   calib_size  : images used to search the quantizer scales (--calibrate)
#   optim_size  : images used for block reconstruction (--optimize)
#   temp        : softmax temperature of the KL loss whose gradients form the GPR basis
#   drop_prob   : QDrop probability (1.0 disables QDrop)
#   alpha       : number of sampled gradients in the GPR basis
#   lambda_max  : final weight of L_SGR
#   sgr_warmup  : fraction of iterations before L_SGR is switched on
class Config:
    def __init__(self):
        # data
        self.calib_size = 128
        self.optim_size = 1024
        self.calib_batch_size = 32
        self.optim_batch_size = 32
        # quantization
        self.w_bit = 6
        self.a_bit = 6
        # block reconstruction
        self.temp = 20
        self.drop_prob = 0.5
        # GPFA-Q
        self.alpha = 32
        self.lambda_max = 0.5
        self.sgr_warmup = 0.2
