import albumentations
import numpy as np
import config as panda_config


class PandaAugmentor:
    """Panda augmentor."""

    def __init__(self,
                 augmentations=None,
                 train_mode=True,
                 dtype='float32'):
        self.augmentations = augmentations
        self.train_mode = train_mode
        self.dtype = dtype
        if self.augmentations is not None:
            self._augmentations = self.augmentations.copy()
            self._normalize = self._augmentations.pop('normalize', None)
        else:
            self._augmentations = None
            self._normalize = None

        # check if normalizing by data provider
        self._norm_by_provider = False
        if self._normalize is not None:
            if self._normalize.get('mean'):
                self._normalize['mean'] = np.array(self._normalize['mean'], dtype=self.dtype)
                self._normalize['std'] = np.array(self._normalize['std'], dtype=self.dtype)
            else:
                self._norm_by_provider = True
                for provider in ['karolinska', 'radboud']:
                    self._normalize[provider]['mean'] = np.array(self._normalize[provider]['mean'], dtype=self.dtype)
                    self._normalize[provider]['std'] = np.array(self._normalize[provider]['std'], dtype=self.dtype)

        self.transform = self._set_transforms()

    def _transform(self, img, data_provider=None):
        if self._norm_by_provider:
            img = (img - self._normalize[data_provider]['mean']) / self._normalize[data_provider]['std']
        elif self._normalize is not None:
            img = (img - self._normalize['mean']) / self._normalize['std']
        return self.transform(image=img.astype(self.dtype))['image']

    def _set_transforms(self):
        transforms = []
        for k, v in self._augmentations.items():
            transforms.append(panda_config.AUGMENTATION_MAP[k](**v))
        return albumentations.Compose(transforms)

    def __call__(self, img, data_provider=None):
        return self._transform(img, data_provider)
