import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex import amp
except:
    pass

from sklearn.metrics import cohen_kappa_score, log_loss
from utils.train_utils import moving_average
from utils.generic_utils import Logger
from utils.model_utils import load_model
from evaluation.metrics import compute_qwk


class Trainer:
    """Panda trainer."""

    def __init__(self,
                 model,
                 optim,
                 criterion,
                 task='clf',
                 scheduler=None,
                 postprocessor=None,
                 max_norm=None,
                 ckpt_dir=None,
                 monitor=None,
                 fp_16=False,
                 rank=0,
                 **kwargs):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.scheduler = scheduler
        self.postprocessor = postprocessor
        self._task = task
        self.ckpt_dir = ckpt_dir
        self._monitor = monitor
        self._rank = rank
        self._max_norm = max_norm
        self._fp_16 = fp_16
        self._is_cuda = next(self.model.parameters()).is_cuda

        self._global_step = None
        self._best_loss = None
        self._best_score = None
        self._best_loss_step = None
        self._best_score_step = None
        self._new_best = False
        self._history = None
        self.logger = None

        if self.logger is None:
            self.logger = sys.stdout

        self._reset()

    @property
    def best_step(self):
        return self._best_score_step

    @property
    def history(self):
        return self._history

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def device(self):
        return next(self.model.parameters()).device

    @device.setter
    def device(self, device):
        self._is_cuda = getattr(device, 'type', device) == 'cuda'

    @property
    def global_step(self):
        self._global_step

    @global_step.setter
    def global_step(self, step):
        self._global_step = step

    def _reset(self):
        if self.ckpt_dir is not None and not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self._history = {
            'loss': [],
            'kappa_score': [],
            'val_loss': [],
            'val_kappa_score': [],
            'elapsed_time': [],
            'lr': [],
        }
        self._empty_cache()
        if self.ckpt_dir is not None:
            self.logger = Logger(os.path.join(self.ckpt_dir, 'train_history.log'))
        else:
            self.logger = sys.stdout

        self._global_step = 0
        self._best_loss = np.inf
        self._best_score = -np.inf
        self._best_loss_step = 0
        self._best_score_step = 0
        self._new_best = False

    def _empty_cache(self):
        self.optim.zero_grad()
        torch.cuda.empty_cache()

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_history(self):
        with open(os.path.join(self.ckpt_dir, 'history.json'), 'w') as f:
            json.dump(str(self._history), f)

    def _save_model(self):
        fname = f"ckpt_{self._global_step:04d}"
        fname += f"_{self._history['val_loss'][-1]:.6f}"
        fname += f"_{self._history['val_kappa_score'][-1]:.6f}.pt"
        filepath = os.path.join(self.ckpt_dir, fname)
        if self._fp_16:
            torch.save({
                'epoch': self._global_step,
                'state_dict': self.model.module.state_dict(),
                'best_loss': self._best_loss,
                'best_score': self._best_score
            }, filepath)
        elif isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), filepath)
        else:
            torch.save(self.model.state_dict(), filepath)

        if self.postprocessor is not None:
            filename = f'coef_{self._global_step:04d}.npy'
            np.save(os.path.join(self.ckpt_dir, filename), self.postprocessor.get_coef())

    def _load_model(self, step=None, filename=None, **kwargs):
        # this needs to be updated with the device
        self.model = load_model(self.ckpt_dir, step, filename, **kwargs)

    def fit(self, train_dl, steps=1, val_dl=None):
        start_time = time.time()
        self._empty_cache()
        if val_dl is not None:
            if isinstance(val_dl, list):
                y_val = torch.Tensor(val_dl[0].dataset.get_labels()).long()
            else:
                y_val = torch.Tensor(val_dl.dataset.get_labels()).long()
            if self._task == 'reg':
                y_val = y_val[...,None].float()

        for step in range(steps):
            self._global_step += 1
            self.train(train_dl)

            str_format = f"step {self._global_step} ({self._history['elapsed_time'][-1]}s) - loss : {self._history['loss'][-1]:.4f}"
            str_format += f" - kappa_score : {self._history['kappa_score'][-1]:.4f}"

            if self._rank == 0:
                if val_dl is not None:
                    y_pred = self.predict(val_dl)
                    val_loss = self.criterion(y_pred, y_val).item()
                    y_pred = y_pred.data.cpu()

                    if self.postprocessor is not None:
                        self.postprocessor.fit(y_pred, y_val.data.cpu().numpy().astype(int))

                    y_pred = self.postprocess(y_pred)
                    val_score = compute_qwk(y_val, y_pred)

                    self._history['val_loss'].append(val_loss)
                    self._history['val_kappa_score'].append(val_score)
                    str_format += f' - val_loss : {val_loss:.4f} - val_kappa_score : {val_score:.4f}'

                    if self.scheduler is not None:
                        str_format += f" - lr : {self.optim.param_groups[0]['lr']:.8f}"

                    best_str = ''
                    if val_loss < self._best_loss:
                        best_str += f'`val_loss` improved from {self._best_loss:.6f} to {val_loss:.6f}'
                        self._best_loss = val_loss
                        self._best_loss_step = step + 1
                        self._new_best = True
                    if val_score > self._best_score:
                        if len(best_str) > 0:
                            best_str += ' - '
                        best_str += f'`val_kappa_score` improved from {self._best_score:.6f} to {val_score:.6f}'
                        self._best_score = val_score
                        self._best_score_step = step + 1
                        self._new_best = True

                    if self._new_best and self.ckpt_dir is not None and self._rank == 0:
                        str_format = best_str + '\n' + str_format
                        self._save_model()

            # clear the current line of the console and log the steps results
            self.logger.write("\033[K")
            self.logger.write(str_format)
            self._new_best = False
            self._history['lr'].append(self.optim.param_groups[0]['lr'])

            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == 'OneCycleLR':
                    pass
                elif self._monitor is None:
                    self.scheduler.step()
                elif self._monitor == 'val_loss':
                    self.scheduler.step(val_loss)
                elif self._monitor == 'val_kappa':
                    self.scheduler.step(val_score)
                else:
                    raise ValueError(f'Unrecognized `monitor` {self._monitor}')

        template_str = f'\n\nFinished training ({int(time.time()-start_time)}s)'
        if val_dl is not None:
            template_str += f' - best val_loss : {self._best_loss:.6f} (step {self._best_loss_step})'
            template_str += f' - best kappa_score : {self._best_score:.6f} (step {self._best_score_step})'
        self.logger.write(template_str)

        if self.ckpt_dir is not None and self._rank == 0:
            self._save_history()
        self._set_eval_mode()

    def train_on_batch(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)

        self.optim.zero_grad()
        if self._fp_16:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
            if self._max_norm:
                nn.utils.clip_grad_norm_(amp.master_params(self.optim), self._max_norm)
        else:
            loss.backward()
            if self._max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._max_norm)

        self.optim.step()
        torch.cuda.synchronize()

        if self.scheduler.__class__.__name__ == 'OneCycleLR':
            if self._fp_16 and amp._amp_state.loss_scalers[0]._unskipped != 0:
                self.scheduler.step()
            else:
                self.scheduler.step()

        y_pred = self.postprocess(y_pred.data.cpu().detach())
        return y_pred, loss.item()

    def postprocess(self, y_pred, from_logit=True):
        if self._task == 'clf':
            return F.softmax(y_pred, dim=1).argmax(1).numpy()
        else:
            if self.postprocessor is not None:
                y_pred = self.postprocessor.predict(y_pred)
            return y_pred

    def train(self, train_dl):
        start_time = time.time()
        self._set_train_mode()
        num_batches = len(train_dl)
        pad_batch = int(np.log10(num_batches))
        train_score = None
        train_loss = None

        train_score_sum = 0
        train_loss_sum = 0
        for i, (x, y) in enumerate(train_dl):
            if self._task == 'reg':
                y = y[...,None].float()
            if self._is_cuda:
                x = x.cuda(self.device)
                y = y.cuda(self.device)

            y_pred, loss = self.train_on_batch(x, y)
            score = compute_qwk(y.data.cpu(), y_pred)
            train_score_sum += score
            train_loss_sum += loss

            template_str = f'step {self._global_step} [{i+1:0{pad_batch+1}d}/{num_batches}'
            template_str += f' ({int(time.time()-start_time)}s)'
            template_str += f' - loss : {train_loss_sum/(i+1):.4f}'
            template_str += f' - kappa_score : {train_score_sum/(i+1):.4f}\r'
            sys.stdout.write(template_str)
            sys.stdout.flush()

        if self._is_cuda:
            torch.cuda.empty_cache()

        self._history['loss'].append(train_loss_sum / num_batches)
        self._history['kappa_score'].append(train_score_sum / num_batches)
        self._history['elapsed_time'].append(int(time.time()-start_time))

    def predict(self, dl):
        self._set_eval_mode()
        preds = []
        num_batches = len(dl)

        with torch.no_grad():
            for i, x in enumerate(dl):
                if isinstance(x, (list, tuple)) and len(x) == 2:
                    x = x[0]
                if self._is_cuda:
                    x = x.cuda(self.device)
                y_pred = self.predict_on_batch(x)
                preds.append(y_pred)

        if self._is_cuda:
            torch.cuda.empty_cache()

        return torch.cat(preds)

    def predict_on_batch(self, x):
        return self.model(x).data.cpu()

    def load_model(self, step=None, filename=None, **kwargs):
        self._load_model(step=step, filename=filename, **kwargs)

    def save_model(self):
        if self.ckpt_dir is not None:
            self._save_model()


class BlendTrainer(Trainer):
    """Blender."""

    def __init__(self,
                 model,
                 optim,
                 criterion,
                 **kwargs):
        super(BlendTrainer, self).__init__(model, optim, criterion, **kwargs)

    def train(self, train_dl):
        start_time = time.time()
        self._set_train_mode()
        num_batches = len(train_dl[0])
        pad_batch = int(np.log10(num_batches))
        train_score = None
        train_loss = None

        train_score_sum = 0
        train_loss_sum = 0
        i = 0
        for batches in zip(*train_dl):
            x = []
            y = []
            for idx, batch_data in enumerate(batches):
                if idx == 0:
                    y = batch_data[1]
                if self._is_cuda:
                    x.append(batch_data[0].cuda(self.device))
                    if idx == 0:
                        y = y.cuda()
                else:
                    x.append(batch_data[0])
                if idx == 0:
                    if self._task == 'reg':
                        y = y[...,None].float()

            y_pred, loss = self.train_on_batch(x, y)
            score = compute_qwk(y.data.cpu(), y_pred)
            train_score_sum += score
            train_loss_sum += loss

            template_str = f'step {self._global_step} [{i+1:0{pad_batch+1}d}/{num_batches}'
            template_str += f' ({int(time.time()-start_time)}s)'
            template_str += f' - loss : {train_loss_sum/(i+1):.4f}'
            template_str += f' - kappa_score : {train_score_sum/(i+1):.4f}\r'
            sys.stdout.write(template_str)
            sys.stdout.flush()
            i += 1

        if self._is_cuda:
            torch.cuda.empty_cache()

        self._history['loss'].append(train_loss_sum / num_batches)
        self._history['kappa_score'].append(train_score_sum / num_batches)
        self._history['elapsed_time'].append(int(time.time()-start_time))

    def predict(self, dl):
        self._set_eval_mode()
        preds = []
        num_batches = len(dl[0])
        for xa, xb in zip(*dl):
            if isinstance(xa, (list, tuple)) and len(xa) == 2:
                xa = xa[0]
            if isinstance(xb, (list, tuple)) and len(xb) == 2:
                xb = xb[0]
            if self._is_cuda:
                xa = xa.cuda(self.device)
                xb = xb.cuda(self.device)
            y_pred = self.predict_on_batch([xa, xb])
            preds.append(y_pred)

        if self._is_cuda:
            torch.cuda.empty_cache()

        return torch.cat(preds)

    def predict(self, dl):
        self._set_eval_mode()
        preds = []
        num_batches = len(dl[0])

        with torch.no_grad():
            for batches in zip(*dl):
                x = []
                for x_batch in batches:
                    if isinstance(x_batch, (list, tuple)) and len(x_batch) == 2:
                        x_batch = x_batch[0]
                    if self._is_cuda:
                        x_batch = x_batch.cuda(self.device)
                    x.append(x_batch)
                y_pred = self.predict_on_batch(x)
                preds.append(y_pred)

        if self._is_cuda:
            torch.cuda.empty_cache()
        return torch.cat(preds)
