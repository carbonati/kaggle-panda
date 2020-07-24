import os
import skimage.io
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from utils import data_utils


class PandaDataset(Dataset):
    """Panda dataset."""
    def __init__(self,
                 root,
                 df,
                 img_size,
                 num_tiles=36,
                 resolution_idx=1,
                 pad_mode=0,
                 th=0.01,
                 method='concat',
                 max_tiles=None,
                 target_col='isup_grade',
                 tile_augmentor=None,
                 img_augmentor=None,
                 meta_cols=None,
                 sample_random=False,
                 replace=True,
                 fp_16=False,
                 seed=None):
        self.root = root
        self.df = df
        self.img_size = img_size
        self.num_tiles = num_tiles
        self.resolution_idx = resolution_idx
        self.pad_mode = pad_mode
        self.th = th
        self.method = method
        self.max_tiles = max_tiles or self.num_tiles
        self.target_col = target_col
        self.tile_augmentor = tile_augmentor
        self.img_augmentor = img_augmentor
        self.meta_cols = meta_cols
        self.sample_random = sample_random
        self.replace = replace
        self._seed = seed
        self._fp_16 = fp_16
        self._dtype = 'float16' if self._fp_16 else 'float32'
        self._dtype_torch = torch.float16 if self._fp_16 else torch.float32
        self._random_state = np.random.RandomState(self._seed)
        self.image_ids = self.df['image_id'].values.tolist()
        if self.target_col is not None:
            self.labels = df['isup_grade'].values.tolist()
            self.training = True
        else:
            self.labels = None
            self.training = False

        self.files = None
        self.image_id_to_filepath = None

        self._set_image_id_to_filepaths()
        self._prepare_args()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.training:
            max_tiles = self.df['num_tiles'].iloc[index]
            num_tiles = min(max_tiles, self.num_tiles) if self.num_tiles is not None else max_tiles
        else:
            num_tiles = self.num_tiles

        n = self._n or int(np.ceil(np.sqrt(num_tiles)))
        remainder = n**2 - max_tiles
        replace = remainder > 0 and self.replace

        if self.sample_random:
            if replace:
                indices = self._random_state.choice(range(max_tiles),
                                                    size=n**2,
                                                    replace=True)
            else:
                indices = self._random_state.choice(range(max_tiles),
                                                    size=n**2-remainder,
                                                    replace=False)

        else:
            indices = list(range(max_tiles))
            if replace:
                for _ in range(int(np.ceil(remainder / max_tiles))):
                    indices += list(range(max_tiles))

        # need to come back to weighted sampling
        if self.training:
            tiles = data_utils.load_tiles(self.root,
                                          self.image_ids[index],
                                          indices=indices,
                                          dtype=self._dtype)
        else:
            tiles = data_utils.load_test_tiles(self.root,
                                    self.image_ids[index],
                                    img_size=self.img_size,
                                    num_tiles=self.num_tiles,
                                    resolution_idx=self.resolution_idx,
                                    pad_mode=self.pad_mode,
                                    th=self.th)
        data_provider = self.df['data_provider'].iloc[index]

        if remainder > 0:
            _indices = list(indices) + [None] * remainder
            indices = self._random_state.choice(_indices, size=n**2, replace=False)

        if self.method == 'concat':
            img = np.zeros((self._h * n, self._w * n, self._c), dtype=self._dtype)
            for i in range(n):
                for j in range(n):
                    idx = indices[i * n + j]
                    if idx is not None:
                        tile = self.preprocess_tile(tiles[idx], data_provider=data_provider)
                        img[i*self._h:(i+1)*self._h, j*self._w:(j+1)*self._w] = tile

            img = self.preprocess_img(img)
            img = torch.tensor(img, dtype=self._dtype_torch).permute(2, 0, 1)
        else:
            img = []
            for i in range(n):
                for j in range(n):
                    idx = indices[i * n + j]
                    if idx is not None:
                        img.append(torch.tensor(tile, dtype=self._dtype_torch))
                    else:
                        img.append(torch.zeros((self.img_size, self.img_size, 3), dtype=self._dtype_torch))

            img = torch.stack(img, 0)

        if self.training:
            return img, self.labels[index]
        else:
            return img

    def _prepare_args(self):
        self._h, self._w = self.img_size, self.img_size
        self._c = 3
        self._n = int(np.sqrt(self.num_tiles)) if self.num_tiles else 0

    def _set_image_id_to_filepaths(self):
        self.image_id_to_filepath = defaultdict(list)
        for image_id in self.image_ids:
            self.image_id_to_filepath[image_id] = os.path.join(self.root, f'{image_id}.tiff')

    def _set_files(self):
        self.files = [fp for fps in self.image_id_to_filepaths.values() for fp in fps]

    def preprocess_tile(self, tile, **kwargs):
        tile = (255 - np.asarray(tile)) / 255
        if self.tile_augmentor is None:
            return tile
        else:
            return self.tile_augmentor(tile, **kwargs)

    def preprocess_img(self, img):
        if self.img_augmentor is None:
            return img
        else:
            return self.img_augmentor(img)

    def get_labels(self):
        return self.labels

    def get_image_ids(self):
        return self.image_ids

    def reset_state(self, seed=None):
        # for bagging + multiprocessing
        self._random_state = np.random.RandomState(seed)
