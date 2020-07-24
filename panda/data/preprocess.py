import os
import numpy as np
import json
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import PandaDataset


def get_tiles(img,
              mask=None,
              img_size=128,
              num_tiles=None,
              pad_mode=0,
              constant_value=255,
              th=0.01,
              return_meta=False):
    result = []
    h, w, c = img.shape
    if w > h:
        img = img.swapaxes(0, 1)
        h, w, c = img.shape
        if mask is not None:
            mask = mask.swapaxes(0, 1)

    pad_h = (img_size - h % img_size) % img_size + ((img_size * pad_mode) // 2)
    pad_w = (img_size - w % img_size) % img_size + ((img_size * pad_mode) // 2)
    img = np.pad(img,
                 [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]],
                 constant_values=constant_value)
    img = img.reshape(
        img.shape[0] // img_size,
        img_size,
        img.shape[1] // img_size,
        img_size,
        3
    )
    img = img.transpose(0,2,1,3,4).reshape(-1, img_size, img_size, 3)

    if mask is not None:
        mask = np.pad(mask,
                      [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]],
                      constant_values=0)
        mask = mask.reshape(
            mask.shape[0] // img_size,
            img_size,
            mask.shape[1] // img_size,
            img_size,
            3
        )
        mask = mask.transpose(0,2,1,3,4).reshape(-1, img_size, img_size, 3)

    img_norm = (constant_value-img).astype(np.float32) / constant_value if constant_value > 0 else img
    x_tissue = img_norm.reshape(len(img), -1, 3).mean(1).mean(-1)

    idxs = np.argsort(-x_tissue)
    idxs_th = idxs[(x_tissue > th)[idxs]]
    if len(idxs_th) == 0:
        idxs_th = idxs[:1]
    if num_tiles is not None:
        idxs_th = idxs_th[:num_tiles]
        num_tiles = len(idxs_th)
    else:
        num_tiles = len(idxs_th)

    img = img[idxs_th]
    if mask is not None:
        mask = mask[idxs_th]

    tiles = [img[i] for i in range(num_tiles)]
    if mask is not None:
        masks = [mask[i] for i in range(num_tiles)]
    if return_meta:
        if len(img) > 0:
            meta = [{'mean': list(x.mean(axis=0)), 'std': list(x.std(axis=0))} for x in img_norm[idxs_th].reshape(num_tiles, -1, 3)]
        else:
            meta = [{'mean': [0.,0.,0], 'std': [0., 0., 0.]}]
        if mask is None:
            return tiles, meta
        else:
            return tiles, masks, meta
    else:
        if mask is None:
            return tiles
        else:
            return tiles, masks


def generate_img_stats(df,
                       root,
                       batch_size=1,
                       num_workers=0,
                       output_dir=None,
                       **kwargs):
    """Generates summary statistics on a dataset."""
    img_ds = PandaDataset(root, df, **kwargs)
    img_dl = DataLoader(img_ds,
                        batch_size=batch_size,
                        num_workers=num_workers)

    img_mean = 0
    img_var = 0
    num_samples = len(img_dl.dataset)

    for i, (x, y) in tqdm(enumerate(img_dl),
                          total=len(img_dl),
                          desc='Generating image stats'):
        x_batch = x.view(x.shape[0], 3, -1)
        img_mean += x_batch.mean(-1).sum(0)
        img_var += x_batch.var(-1).sum(0)

    img_mean /= num_samples
    img_var /= num_samples
    img_stats = {
        'mean': list(np.asarray(img_mean)),
        'std': list(np.asarray(np.sqrt(img_var))),
        'max_pixel_value': 1
    }

    # save image stats to disk
    if output_dir is not None:
        output = {}
        for k, v in img_stats.items():
            if isinstance(v, torch.Tensor):
                output[k] = list(v.numpy())
            else:
                output[k] = v
        with open(os.path.join(output_dir, 'img_stats.json'), 'w') as f:
            json.dump(str(output), f)

    return img_stats


def remove_pen_marks(img):
    """https://www.kaggle.com/akensert/panda-removal-of-pen-marks"""
    # Define elliptic kernel
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # use cv2.inRange to mask pen marks (hardcoded for now)
    lower = np.array([0, 0, 0])
    upper = np.array([200, 255, 255])
    img_mask1 = cv2.inRange(img, lower, upper)

    # Use erosion and findContours to remove masked tissue (side effect of above)
    img_mask1 = cv2.erode(img_mask1, kernel5x5, iterations=4)
    img_mask2 = np.zeros(img_mask1.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(img_mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        w, h = x.max()-x.min(), y.max()-y.min()
        if w > 100 and h > 100:
            cv2.drawContours(img_mask2, [contour], 0, 1, -1)
    # expand the area of the pen marks
    img_mask2 = cv2.dilate(img_mask2, kernel5x5, iterations=3)
    img_mask2 = (1 - img_mask2)

    # Mask out pen marks from original image
    img = cv2.bitwise_and(img, img, mask=img_mask2)

    img[img == 0] = 255

    return img
