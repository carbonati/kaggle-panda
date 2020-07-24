import os
import sys
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import time
import datetime
import argparse
import torch
import torch.nn as nn
warnings.simplefilter("ignore", UserWarning)
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel, convert_syncbn_model
except:
    pass

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from panda.utils import generic_utils as utils
from panda.utils import train_utils, model_utils
from panda.data.dataset import PandaDataset
from panda.core.trainer import BlendTrainer
from panda.core.models import BlendModel
from panda.evaluation.postprocess import get_branch, generate_df_pred, log_model_summary
import panda.config as panda_config


def blend(config):
    """Run a panda training session."""
    # path to checkpoint models to blend
    if config.get('ckpt_dirs'):
        ckpt_dirs = config['ckpt_dirs']
    else:
        base_dir = os.path.join(config['input']['models'], config['experiment_name'])
        ckpt_dirs = [os.path.join(base_dir, fn) for fn in os.listdir(base_dir)]
    num_models = len(ckpt_dirs)

    # clean up the model directory and generate a new output path for the training session
    log_dir = os.path.join(config['output']['models'], config['experiment_name'])
    model_fname = f"blend_{num_models}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    model_dir = os.path.join(log_dir, model_fname)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # log activity from the training session to a logfile
    #if config['local_rank'] == 0:
    #    sys.stdout = utils.Tee(os.path.join(model_dir, 'train_history.log'))
    utils.set_state(config['random_state'])
    device_ids = config.get('device_ids', [0])
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # cuda settings
    if config['distributed']:
        torch.cuda.set_device(config['local_rank'])
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    else:
        if isinstance(device_ids, list):
            visible_devices = ','.join([str(x) for x in device_ids])
        else:
            visible_devices = device_ids
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    fold_ids = config['fold_ids']
    fold_ids = fold_ids if isinstance(fold_ids, list) else [fold_ids]

    # build each dataset and model to blend
    for fold in fold_ids:
        fold = int(fold) if fold.isdigit() else fold
        models = []
        train_dls = []
        val_dls = []
        test_dls = []
        for ckpt_dir in ckpt_dirs:
            model, train_dl, val_dl, test_dl = get_branch(ckpt_dir=ckpt_dir,
                                                          root=config['input']['root'],
                                                          cv_folds_dir=config['input']['cv_folds'],
                                                          train=config['input']['train'],
                                                          fold=fold,
                                                          patient=config['input'].get('patient'),
                                                          blacklist=config['input'].get('blacklist'),
                                                          batch_size=config['batch_size'],
                                                          num_workers=config['num_workers'],
                                                          keep_prob=config.get('keep_prob', 1),
                                                          verbose=config.get('verbose', 1))
            models.append(model)
            train_dls.append(train_dl)
            val_dls.append(val_dl)
            test_dls.append(test_dl)

        model = BlendModel(models, **config['model'].get('params', {}))
        if config['input'].get('pretrained'):
            print(f"Loading pretrained weights from {config['input']['pretrained']}")
            model.load_state_dict(torch.load(config['input']['pretrained']))

        model = model.cuda()
        if config['distributed']:
            model = convert_syncbn_model(model)
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = nn.DataParallel(model, device_ids).to(device)

        optim = train_utils.get_optimizer(model=model, **config['optimizer'])
        sched = train_utils.get_scheduler(config,
                                          optim,
                                          steps_per_epoch=len(train_dl))
        criterion = train_utils.get_criterion(config)
        postprocessor = train_utils.get_postprocessor(**config['postprocessor']) if config.get('postprocessor') else None

        if config['distributed'] and config['fp_16']:
            model, optim = amp.initialize(model,
                                          optim,
                                          opt_level=config.get('opt_level', 'O2'),
                                          loss_scale='dynamic',
                                          keep_batchnorm_fp32=config.get('keep_batchnorm_fp32', True))

        ckpt_dir = os.path.join(model_dir, f'fold_{fold}')
        print(f'Checkpoint path {ckpt_dir}\n')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        trainer = BlendTrainer(model,
                               optim,
                               criterion,
                               postprocessor=postprocessor,
                               scheduler=sched,
                               ckpt_dir=ckpt_dir,
                               fp_16=config['fp_16'],
                               r2ank=config.get('local_rank', 0),
                               **config.get('trainer', {}))
        trainer.fit(train_dls, config['steps'], val_dls)

        # generate predictions table from the best step model
        if config['local_rank'] == 0:
            trainer.load_model(step=trainer.best_step, models=models)
            df_pred_val = generate_df_pred(trainer,
                                           val_dls,
                                           val_dls[0].dataset.df,
                                           postprocessor=postprocessor,
                                           mode='blend')
            df_pred_val.to_csv(os.path.join(ckpt_dir, 'val_predictions.csv'), index=False)
            log_model_summary(df_pred_val, logger=trainer.logger, group='val')
            if postprocessor is not None:
                print('Updating postprocessor val predictions')
                postprocessor.fit(df_pred_val['prediction_raw'], df_pred_val[config['target_col']])

            if test_dl is not None:
                df_pred_test = generate_df_pred(trainer,
                                                test_dls,
                                                test_dls[0].dataset.df,
                                                postprocessor=postprocessor,
                                                mode='blend',
                                                num_bags=config.get('num_bags'))
                df_pred_test.to_csv(os.path.join(ckpt_dir, 'test_predictions.csv'), index=False)
                log_model_summary(df_pred_test, logger=trainer.logger, group='test')

            if postprocessor is not None:
                np.save(os.path.join(ckpt_dir, 'coef.npy'), postprocessor.get_coef())
            print(f'Saved output to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        type=str,
                        help='Path to configurable file.')
    parser.add_argument('--local_rank', '-r', default=0, type=int)
    parser.add_argument('--distributed', '-d', default=False, action="store_true")
    parser.add_argument('--fp_16', '-fp16', default=False, action="store_true")
    parser.add_argument('--steps', '-s', default=None, type=int)
    parser.add_argument('--keep_prob', '-p', default=None, type=float)
    parser.add_argument('--batch_size', '-bs', default=None, type=int)
    parser.add_argument('--experiment_name', '-e', default=None, type=str)
    parser.add_argument('--num_workers', '-w', default=None, type=int)
    parser.add_argument('--fold_ids', default=None, type=list)
    parser.add_argument('--num_gpus', default=1, type=int)
    args = parser.parse_args()

    config = utils.load_config_from_yaml(args.config_filepath)
    config = utils.prepare_config(config, args)

    blend(config)

