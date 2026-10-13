import os
import yaml
from yacs.config import CfgNode as CN
_C = CN()
_C.BASE = ['']
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4
_C.DATA.DATA_PATH = 'data/Provincial_Capital'
_C.DATA.DATASET = 'ProvincialCapital'
_C.DATA.IMG_SIZE = 1024
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.num_workers = 8
_C.MODEL = CN()
_C.MODEL.PRETRAIN_CKPT = ''
_C.MODEL.NUM_CLASSES = 9
_C.MODEL.DROP_RATE = 0
_C.MODEL.DROP_PATH_RATE = 0.3
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.IGNORE_INDEX = 255
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 8
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 0.0002
_C.TRAIN.WARMUP_LR = 1e-07
_C.TRAIN.MIN_LR = 1e-07
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-08
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.SEED = 0
_C.EVAL_MODE = False
_C.LOCAL_RANK = 0
_C.SAVE_TOP_K = 1
_C.MONITOR = 'val_Target_IoU'
_C.MONITOR_MODE = 'max'
_C.SAVE_LAST = True
_C.WEIGHTS_NAME = 'TargetIoU_best'
_C.LOG_NAME = 'Mine_experiment'
_C.CHECK_VAL_EVERY_N_EPOCH = 1
_C.gpus = 1
_C.OUTPUT = 'results'
_C.MODEL.NAME = 'UrbanSeg'

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    config.freeze()

def get_config(args):
    config = _C.clone()
    config.set_new_allowed(True)
    update_config(config, args)
    return config
