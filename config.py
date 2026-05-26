import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 4
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = './data/Provincial_Capital'
# Dataset name
_C.DATA.DATASET = 'ProvincialCapital'
# Input image size
_C.DATA.IMG_SIZE = 1024
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads (recommended: 4 or 8 for faster data loading)
_C.DATA.num_workers = 8
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.PRETRAIN_CKPT = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 9
# Dropout rate
_C.MODEL.DROP_RATE = 0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.3
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
# Loss ignore index (default: 255)
_C.MODEL.IGNORE_INDEX = 255

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 8
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 2e-4
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-7
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 1.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

# --- Checkpoint and Log Settings ---
_C.SAVE_TOP_K = 1
_C.MONITOR = 'val_Target_IoU'
_C.MONITOR_MODE = 'max'
_C.SAVE_LAST = True
_C.WEIGHTS_NAME = 'TargetIoU_best'
_C.LOG_NAME = 'Mine_experiment'
_C.CHECK_VAL_EVERY_N_EPOCH = 1
_C.gpus = 1
_C.OUTPUT = r'.results'
_C.MODEL.NAME = 'UrbanSeg'

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    config.defrost()

    if isinstance(args, str):
        _update_config_from_file(config, args)
    else:
        if hasattr(args, 'opts') and args.opts:
            config.merge_from_list(args.opts)
        if hasattr(args, 'batch_size') and args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    config.set_new_allowed(True)
    update_config(config, args)
    return config
