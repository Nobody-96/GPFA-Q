# GPFA-Q entry point: calibration, block reconstruction and evaluation.
import os
import sys
import argparse
import importlib
import logging
import random
import builtins
from datetime import datetime

import numpy as np
import torch
from torch import nn
import timm

from utils.datasets import ViTImageNetLoaderGenerator
from utils.calibrator import QuantCalibrator
from utils.block_recon_gpfaq import GPFAQBlockReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import validate

original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

# Part of the reproducibility setup: some reductions run on the CPU.
torch.set_num_threads(2)

OUTPUT_ROOT = './results'
DEVICE = 'cuda'
NUM_WORKERS = 8
PRINT_FREQ = 10

MODEL_ZOO = {
    'vit_small':  'vit_small_patch16_224',
    'vit_base':   'vit_base_patch16_224',
    'deit_tiny':  'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base':  'deit_base_patch16_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base':  'swin_base_patch4_window7_224',
}

# --ablation -> (use GPR, use SGR), following Table 2 of the paper.
ABLATIONS = {
    'gpfaq':    (True,  True),    # Baseline + GPR + SGR (full method)
    'gpr':      (True,  False),   # Baseline + GPR
    'baseline': (False, False),   # Baseline
}


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deit_small', choices=list(MODEL_ZOO.keys()),
                        help='model')
    parser.add_argument('--config', type=str, required=True,
                        help='file to import the Config class from, e.g. ./configs/4bit/gpfaq.py')
    parser.add_argument('--dataset', type=str, required=True,
                        help='ImageNet root that contains train/ and val/')

    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help='calibrate the model')
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None,
                                      help='path to a calibrated checkpoint')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true',
                                     help='run GPFA-Q block reconstruction')
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None,
                                     help='path to a reconstructed checkpoint')
    parser.add_argument('--test-optimize-checkpoint', action='store_true',
                        help='validate the loaded reconstructed checkpoint')

    parser.add_argument('--ablation', type=str, default='gpfaq', choices=list(ABLATIONS.keys()),
                        help='gpfaq = Baseline+GPR+SGR (default), gpr = Baseline+GPR, baseline = Baseline')
    parser.add_argument('--seed', default=3407, type=int, help='seed')
    parser.add_argument('--val-batch-size', default=200, type=int, help='batch size for validation')
    return parser


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_run_dir(args, cfg):
    if args.optimize:
        tag = args.ablation
    elif args.calibrate:
        tag = 'calib'
    else:
        tag = 'eval'
    return os.path.join(OUTPUT_ROOT, f'{args.model}_w{cfg.w_bit}_a{cfg.a_bit}', f'seed{args.seed}', tag)


def save_model(model, args, cfg, run_dir, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_mse.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size)
    else:
        auto_name = '{}_w{}_a{}_optimsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, args.ablation)
    save_path = os.path.join(run_dir, auto_name)
    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    ckpt = torch.load(ckpt_path)
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + '.' + attr + '.scale'
                getattr(module, attr).scale.data = ckpt[ckpt_name].clone()
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model


def main(args):
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    cfg = Config()

    run_dir = get_run_dir(args, cfg)
    os.makedirs(run_dir, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(os.path.join(run_dir, 'output.log')),
            logging.StreamHandler()
        ]
    )
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    logging.info(f"Config imported from {args.config}")
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")

    device = torch.device(DEVICE)

    seed_all(args.seed)

    # The first attempt also initialises a model before failing when the file is absent,
    # which advances the RNG. It is kept so that runs reproduce the paper's data sampling.
    logging.info('Building model ...')
    try:
        model = timm.create_model(MODEL_ZOO[args.model],
                                  checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(MODEL_ZOO[args.model]))
    except Exception:
        model = timm.create_model(MODEL_ZOO[args.model], pretrained=True)

    model.to(device)
    model.eval()
    g = ViTImageNetLoaderGenerator(args.dataset, model, args.val_batch_size, NUM_WORKERS)
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wrapping quantization modules (reparam: {}) ...'.format(reparam))
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device)

    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
        else:
            logging.info("{} - start mse guided calibration".format(get_cur_time()))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - mse guided calibration finished.".format(get_cur_time()))
            save_model(model, args, cfg, run_dir, mode='calibrate')
            # Also part of the random stream that a following --optimize continues from.
            logging.info('Validating after calibration ...')
            validate(val_loader, model, criterion, print_freq=PRINT_FREQ, device=device)

    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        use_gpr, use_sgr = ABLATIONS[args.ablation]
        logging.info("{} - start GPFA-Q block reconstruction (ablation={}, GPR={}, SGR={})".format(
            get_cur_time(), args.ablation, use_gpr, use_sgr))
        block_reconstructor = GPFAQBlockReconstructor(
            model, calib_loader, batch_size=cfg.optim_batch_size, temp=cfg.temp,
            alpha=cfg.alpha, lambda_max=cfg.lambda_max, sgr_warmup=cfg.sgr_warmup,
            drop_prob=cfg.drop_prob, use_gpr=use_gpr, use_sgr=use_sgr,
        )
        block_reconstructor.reconstruct_model()
        save_model(model, args, cfg, run_dir, mode='optimize')

    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        validate(calib_loader, model, criterion, print_freq=PRINT_FREQ, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        validate(val_loader, model, criterion, print_freq=PRINT_FREQ, device=device)
    logging.info("{} - finished the process.".format(get_cur_time()))


if __name__ == "__main__":
    main(get_args_parser().parse_args())
