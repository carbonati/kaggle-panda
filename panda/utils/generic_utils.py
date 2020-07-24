import os
import sys
import time
import glob
import yaml
import datetime
import shutil
import random
import pprint
import numpy as np
import torch
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


# generic utils
class Logger(object):
    """Basic logger to record stdout activity"""
    def __init__(self, filename):
        self._filename = filename
        self.terminal = sys.stdout

    def write(self, msg):
        msg += '\n'
        self.terminal.write(msg)
        with open(self._filename, 'a+') as f:
            f.write(msg)

    def carriage(self, msg):
        msg += '\r'
        self.terminal.write(msg)
        with open(self._filename, 'w') as f:
            f.write(msg)
        sys.stdout.flush()


class Tee(object):

    def __init__(self, name, mode='a+'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
         sys.stdout = self.stdout
         self.file.close()

    def write(self, data):
         self.file.write(data)
         self.stdout.write(data)

    def flush(self):
         self.file.flush()


def get_model_fname(config):
    """
    scheme
    ------
    arch
    tile_size
    num_tiles
    resolution_idx
    pad_mode
    batch_size
    pool_method
    optimizer
    scheduler
    description
    datetime
    """
    dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model_fname = f"{config['model']['method']}_{config['model']['params']['backbone']}"
    model_fname += f"_{config['criterion']['method']}"
    model_fname += f"_{config['data']['img_size']}"
    model_fname += f"_{config['data']['num_tiles']}_{config['data']['resolution_idx']}"
    if config['data'].get('pad_mode'):
        model_fname += f"_{config['data']['pad_mode']}"
    if config['data'].get('sample_random'):
        model_fname += "_rand"
    model_fname += f"_{config['batch_size']}"
    if config['model'].get('pool_params'):
        model_fname += f"_{config['model']['pool_params']['method']}"
    model_fname += f"_{config['optimizer']['method']}"
    if config.get('scheduler'):
        model_fname += f"_{config['scheduler']['method']}"
    if config.get('subset'):
        for k, v in config['subset'].items():
            model_fname += f'_{v}'
    if config.get('description'):
        model_fname += f"_{'_'.join(config['description'].split(' '))}"
    model_fname += f"_{dt_str}"
    return model_fname


def prepare_config(config, args):
    for arg in vars(args):
        v = getattr(args, arg)
        if v is not None:
            print(f'Updating parameter `{arg}` to {v}')
            config[arg] = v
    return config


def cleanup_log_dir(root, min_steps=5, keep_last_n=5):
    for model_name in os.listdir(root):
        model_dir = os.path.join(root, model_name)
        cleanup_ckpts(model_dir)


def cleanup_ckpts(model_dir, min_steps=5, keep_last_n=5):
    if not os.path.exists(os.path.join(model_dir, 'train_history.log')):
        return
    fold_dirs = glob.glob(os.path.join(model_dir, 'fold_*'))
    ckpt_files = []
    if len(fold_dirs) > 0:
        for fold_dir in fold_dirs:
            max_step = 0
            for fn in os.listdir(fold_dir):
                if fn.startswith('ckpt_'):
                    step = int(fn.split('_')[1])
                    if step > max_step:
                        max_step = step
                    ckpt_files.append(fn)
            if max_step < min_steps:
                shutil.rmtree(fold_dir)
            else:
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1]))
                if len(ckpt_files) > keep_last_n:
                    for fn in ckpt_files[:-keep_last_n]:
                            os.remove(os.path.join(fold_dir, fn))
        if not any([os.path.exists(f) for f in fold_dirs]):
            shutil.rmtree(model_dir)
    else:
        shutil.rmtree(model_dir)


def set_state(seed=42069):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded configurable file from {filepath}')
    pprint.pprint(config)
    return config


def _compute_num_tiles(image_id, image_id_list):
    return image_id_list.count(image_id)


def compute_num_tiles(image_id_list, num_workers=1):
    image_ids = list(set(image_id_list))
    f = partial(_compute_num_tiles, image_id_list=image_id_list)
    with mp.Pool(num_workers) as pool:
        num_tiles = list(tqdm(pool.map(f, image_ids),
                              total=len(image_ids),
                              desc='Counting image_id tiles'))
    return list(zip(*(image_ids, num_tiles)))

