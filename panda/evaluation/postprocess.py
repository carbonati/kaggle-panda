import os
import numpy as np
import pandas as pd
import time
from functools import partial
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, RandomSampler

from utils import data_utils, model_utils, train_utils
from data.augmentation import PandaAugmentor
from data.dataset import PandaDataset
from evaluation.metrics import compute_qwk, compute_provider_scores


class OptimizedRounder():
    """Modified optimzied rounder.
    Originally proposed in https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107
    """
    def __init__(self,
                 num_classes=6,
                 loss_fn=None,
                 method='nelder-mead',
                 initial_coef=None):
        self._num_classes = num_classes
        self._loss_fn = compute_qwk if loss_fn is None else loss_fn
        if initial_coef is None:
            self._initial_coef = [c-.5 for c in range(1, self._num_classes)]
        else:
            self._initial_coef = initial_coef
        self._method = method
        self.res = None
        self.coef_ = self._initial_coef.copy()

    def _predict(self, X, coef=None):
        coef = self.coef_ if coef is None else coef
        X_p = np.copy(X)

        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        return X_p

    def _compute_loss(self, coef, X, y):
        X_p = self._predict(X, coef=coef)
        ll = self._loss_fn(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_ = partial(self._compute_loss, X=X, y=y)
        self.res = minimize(loss_,
                            self._initial_coef,
                            method=self._method)
        self.coef_ = self.res['x']

    def predict(self, X, coef=None):
        return self._predict(X, coef=None)

    def get_coef(self):
        return self.coef_

    def set_coef(self, coef):
        self.coef_ = coef


def generate_df_pred(trainer,
                     dl,
                     df_panda=None,
                     postprocessor=None,
                     num_classes=6,
                     num_bags=0,
                     mode=None,
                     pred_cols=None):
    """Returns a table of predictions for each sample of in a dataloader."""
    start_time = time.time()
    if pred_cols is None:
        pred_cols = ['image_id', 'isup_grade', 'prediction']

    # load the best model and save validation predictions to disk
    if mode == 'blend':
        y_true = dl[0].dataset.get_labels()
        image_ids = dl[0].dataset.get_image_ids()
    else:
        y_true = dl.dataset.get_labels()
        image_ids = dl.dataset.get_image_ids()

    # generate (bagged) predictions
    if num_bags > 0:
        y_pred_raw = []
        for i in range(num_bags):
            y_pred_raw.append(trainer.predict(dl).numpy())
            if mode == 'blend':
                for sub_dl in dl:
                    sub_dl.dataset.reset_state()
            else:
                dl.dataset.reset_state()
        y_pred_raw = np.mean(y_pred_raw, axis=0)
    else:
        y_pred_raw = trainer.predict(dl).numpy()

    # generate postprocessed predictions
    num_classes = y_pred_raw.shape[1]
    if num_classes > 1:
        pred_cols.extend([f'grade_{i}' for i in range(num_classes)])
        y_pred = y_pred_raw.argmax(1).astype(int)
    else:
        pred_cols.append('prediction_raw')
        if postprocessor is not None:
            y_pred = postprocessor.predict(y_pred_raw)
        else:
            y_pred = y_pred_raw.round()
        y_pred = y_pred.flatten().astype(int)

    # generate prediction table
    pred_data = [
        image_ids,
        y_true,
        y_pred,
        *y_pred_raw.T
    ]
    df_pred = pd.DataFrame(zip(*pred_data), columns=pred_cols)
    if df_panda is not None:
        df_pred = pd.merge(df_pred,
                           df_panda[['image_id', 'data_provider']],
                           how='left',
                           on='image_id')

    print(f'Total time to generate predictions : {int(time.time()-start_time)}s')
    return df_pred


def log_model_summary(df_pred,
                      logger=None,
                      target_col='isup_grade',
                      group='val'):
    logger = logger or sys.stdout
    template_str = f'{group} model scores'
    logger.write('-'*len(template_str))
    logger.write(template_str)
    logger.write('-'*len(template_str))

    qwk = compute_qwk(df_pred[target_col], df_pred['prediction'])
    acc = accuracy_score(df_pred[target_col], df_pred['prediction'])
    logger.write(f'\nQWK : {qwk:.6f}\nACC : {acc:.6f}')

    # log scores by institution
    logger.write(f"\ndata_provider\n{'-'*13}")
    provider_scores = compute_provider_scores(df_pred)
    for k, v in provider_scores.items():
        logger.write(f"{k} \t: {v:.4f}")

    # log confusion matrix
    c_mtx = confusion_matrix(df_pred[target_col], df_pred['prediction'])
    logger.write('\n' + repr(c_mtx) + '\n')


def get_branch(ckpt_dir,
               fold,
               root,
               cv_folds_dir,
               train,
               patient=None,
               blacklist=None,
               marker=None,
               step=None,
               batch_size=8,
               eval_batch_size=None,
               distributed=False,
               num_gpus=1,
               keep_prob=1,
               num_workers=8,
               verbose=1,
               random_state=420):
    """Returns a fine tuned model, train, val, and test dataloaders from a `ckpt_dir`"""
    fold_dir = os.path.join(ckpt_dir, f'fold_{fold}')
    config = model_utils.load_config(ckpt_dir)
    if step is None:
        step = max([int(fn.split('_')[1]) for fn in os.listdir(fold_dir) if fn.startswith('ckpt')])

    tile_version = config['input']['images'].strip('/').split('/')[-1]
    image_dir = os.path.join(root, tile_version, 'tiles')
    num_tiles = os.path.join(root, tile_version, 'num_tiles.csv')
    num_tiles = num_tiles if os.path.exists(num_tiles) else config['data']['num_tiles']
    df_panda = data_utils.load_data(train,
                                    image_dir,
                                    patient=patient,
                                    blacklist=blacklist,
                                    num_tiles=num_tiles,
                                    marker=marker)
    if keep_prob < 1:
        keep_n = int(np.ceil(len(df_panda) * keep_prob))
        df_panda = df_panda.iloc[:keep_n]
    df_panda = df_panda.iloc[:200]
    cv_folds = data_utils.load_cv_folds(os.path.join(cv_folds_dir, 'cv_folds.p'))
    df_panda['fold'] = data_utils.get_fold_col(df_panda, cv_folds)

    img_stats = data_utils.load_img_stats(os.path.join(cv_folds_dir, tile_version), 0)
    tile_aug_params = config['augmentations'].get('tile', {})
    tile_aug_params['normalize'] = img_stats
    img_aug_params = config['augmentations'].get('img', {})
    tile_aug_train, tile_aug_val, tile_aug_test = train_utils.get_augmentors(tile_aug_params)
    img_aug_train, img_aug_val, img_aug_test = train_utils.get_augmentors(img_aug_params)

    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    df_train = df_panda.loc[(df_panda['fold'] != fold) &
                            (df_panda['fold'] != 'test')].reset_index(drop=True)
    df_val = df_panda.loc[df_panda['fold'] == fold].reset_index(drop=True)
    if os.path.exists(os.path.join(fold_dir, 'val_predictions.csv')):
        df_val_pred = pd.read_csv(os.path.join(fold_dir, 'val_predictions.csv'))
        df_val = pd.merge(df_val, df_val_pred[['image_id', 'prediction']], how='left', on='image_id')
    df_test = df_panda.loc[df_panda['fold'] == 'test'].reset_index(drop=True)
    if os.path.exists(os.path.join(fold_dir, 'test_predictions.csv')):
        df_test_pred = pd.read_csv(os.path.join(fold_dir, 'test_predictions.csv'))
        df_test = pd.merge(df_test, df_test_pred[['image_id', 'prediction']], how='left', on='image_id')

    if verbose:
        df_sub_val = df_val.loc[df_val['prediction'].notnull()].reset_index(drop=True)
        df_sub_test = df_test.loc[df_test['prediction'].notnull()].reset_index(drop=True)
        print('-'*80)
        print(ckpt_dir)
        print(f"\nVal  QWK : {compute_qwk(df_sub_val['isup_grade'], df_sub_val['prediction']):.4f}\n")
        print(compute_provider_scores(df_sub_val))
        print(f"\nTest QWK : {compute_qwk(df_sub_test['isup_grade'], df_sub_test['prediction']):.4f}\n")
        print(compute_provider_scores(df_sub_test))
        print('-'*80, '\n')

    train_ds = PandaDataset(image_dir,
                            df_train,
                            tile_augmentor=tile_aug_train,
                            img_augmentor=img_aug_train,
                            **config['data'])
    val_ds = PandaDataset(image_dir,
                          df_val,
                          tile_augmentor=tile_aug_val,
                          img_augmentor=img_aug_val,
                          **config['data'])
    test_ds = PandaDataset(image_dir,
                           df_test,
                           tile_augmentor=tile_aug_test,
                           img_augmentor=img_aug_test,
                           **config['data'])

    train_sampler = train_utils.get_sampler(train_ds,
                                            distributed=distributed,
                                            batch_size=batch_size * num_gpus,
                                            random_state=random_state,
                                            method=config['sampler']['method'],
                                            params=config['sampler'].get('params', {}))
    val_sampler = train_utils.get_sampler(val_ds,
                                          method='sequential',
                                          distributed=distributed)
    test_sampler = train_utils.get_sampler(test_ds,
                                           method='sequential',
                                           distributed=distributed)

    train_dl = DataLoader(train_ds,
                          batch_size=eval_batch_size,
                          num_workers=num_workers,
                          sampler=train_sampler,
                          drop_last=True)
    val_dl = DataLoader(val_ds,
                        batch_size=eval_batch_size,
                        sampler=val_sampler,
                        num_workers=num_workers)
    test_dl = DataLoader(test_ds,
                         batch_size=eval_batch_size,
                         sampler=test_sampler,
                         num_workers=num_workers)

    model = model_utils.load_model(fold_dir, step=step)

    return model, train_dl, val_dl, test_dl
