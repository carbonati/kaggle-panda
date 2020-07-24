import os
import numpy as np
import pandas as pd
import PIL
import skimage.io
import pickle
import json
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from data import preprocess


def get_image_ids(filepath):
    image_filepaths = os.listdir(filepath)
    if image_filepaths[0].endswith('.tiff'):
        split_by = '.tiff'
    elif image_filepaths[0].endswith('.png'):
        split_by = '_'
    else:
        warnings.warn('prob gonna fail')
        split_by = '.'

    image_ids = sorted(set([fn.split(split_by)[0] for fn in image_filepaths]))
    return image_ids


def load_data(train,
              images,
              patient=None,
              blacklist=None,
              subset_dict=None,
              num_tiles=None,
              marker=None,
              min_tiles=0,
              keep_prob=1):
    # read in the training table and generate cv folds
    df_panda = pd.read_csv(train)
    df_panda.set_index('image_id', inplace=True)

    if os.path.exists(os.path.join(images, 'tiles')):
        images = os.path.join(images, 'tiles')
    image_ids = get_image_ids(images)

    # check if we should subset
    if keep_prob < 1 and keep_prob > 0:
        image_ids = sorted(np.random.choice(image_ids, int(len(image_ids) * keep_prob), replace=False))

    df_panda = df_panda.loc[image_ids].reset_index()
    df_panda.loc[df_panda['gleason_score'] == 'negative', 'gleason_score'] = '0+0'
    df_panda[['grade_1', 'grade_2']] = df_panda['gleason_score'].apply(
        lambda x: pd.Series(x.split('+'))
    )

    if patient:
        df_patient = pd.read_csv(patient)
        df_panda = pd.merge(df_panda, df_patient, how='left', on='image_id')

    if marker is not None:
        with open(marker, 'rb') as f:
            marker_image_ids = pickle.load(f)
        df_panda['marker'] = 0
        df_panda.loc[df_panda['image_id'].isin(marker_image_ids), 'marker'] = 1

    # remove blacklisted image_ids (low tissue %)
    if blacklist:
        with open(blacklist, 'rb') as f:
            blacklist = pickle.load(f)
        print(f'removing {len(blacklist)} blacklist image_ids')
        df_panda = df_panda.loc[~df_panda['image_id'].isin(blacklist)].reset_index(drop=True)

    if subset_dict:
        cond = []
        for k, v in subset_dict.items():
            cond.append(df_panda[k] == v)
        df_panda = df_panda.loc[sum(cond) == len(cond)].reset_index(drop=True)

    if isinstance(num_tiles, str) and os.path.exists(num_tiles):
        df_tiles = pd.read_csv(num_tiles)
        df_panda = pd.merge(df_panda, df_tiles, how='left', on='image_id')
        print(f'Dropping slides with < {min_tiles} tiles')
        df_panda = df_panda.loc[(df_panda['num_tiles'].notnull()) &
                                (df_panda['num_tiles'] > min_tiles)].reset_index(drop=True)
        df_panda['num_tiles'] = df_panda['num_tiles'].astype(int)
    else:
        df_panda['num_tiles'] = num_tiles

    return df_panda


def generate_cv_folds(df,
                      test_size=0.1,
                      num_folds=10,
                      train_cols=None,
                      stratify_test=None,
                      stratify_val=None,
                      index_col='patient_id',
                      agg_fnc='max',
                      random_state=42069):

    if train_cols is not None:
        if not isinstance(train_cols, (list, tuple)):
            train_cols = [train_cols]
        train_ids_force = set()
        for col in train_cols:
            train_ids_force.update(df.loc[df[col] == 1, index_col].tolist())
        train_ids_force = list(train_ids_force)
    else:
        train_ids_force = []

    df_fold = df.loc[~df[index_col].isin(train_ids_force)].set_index(index_col)

    if test_size > 0:
        if stratify_test is not None:
            if not isinstance(stratify_test, (tuple, list)):
                stratify_test = [stratify_test]

            df_fold = df_fold.groupby(index_col)[stratify_test].agg(agg_fnc)
            targets = df_fold.apply(
                lambda x: '_'.join([str(x[col]) for col in stratify_test]),
                axis=1
            ).values
        else:
            targets = None

        train_idx, test_idx = train_test_split(df_fold.index,
                                               stratify=targets,
                                               test_size=test_size,
                                               random_state=random_state)
    else:
        train_idx = df_fold.index
        test_idx = []

    # add back train ID's to use for CV
    train_idx = list(train_idx) + train_ids_force

    if stratify_val is not None:
        if not isinstance(stratify_val, (tuple, list)):
            stratify_val = [stratify_val]

        df_train = df.loc[df[index_col].isin(train_idx)]
        df_train = df_train.groupby(index_col)[stratify_val].agg(agg_fnc)
        # update target values to stratify on the updated train set
        targets = df_train.apply(
            lambda x: '_'.join([str(x[col]) for col in stratify_val]),
            axis=1
        ).values
        kf = StratifiedKFold(num_folds,
                             random_state=random_state,
                             shuffle=True)
    else:
        df_train = df.loc[df[index_col].isin(train_idx)].set_index(index_col)
        targets = None
        kf = KFold(num_folds,
                   random_state=random_state,
                   shuffle=True)

    cv_folds = []
    for i, (tr_idx, val_idx) in enumerate(kf.split(df_train.index, y=targets)):
        cv_folds.append(df_train.index[val_idx].tolist())
    # add test id's as the last fold
    cv_folds.append(list(test_idx))
    return cv_folds


def get_fold_col(df, cv_folds, index_col='patient_id'):
    num_folds = len(cv_folds) - 1
    index = df.index if index_col is None else df[index_col]
    s = pd.Series([None] * len(df), index=index)
    for i, val_idx in enumerate(cv_folds):
        if i < num_folds:
            s.loc[s.index.isin(val_idx)] = i
        else:
            s.loc[s.index.isin(val_idx)] = 'test'
    return s.values


def load_cv_folds(filepath):
    with open(filepath, 'rb') as f:
        cv_folds = pickle.load(f)
    return cv_folds


def load_tiles(root, image_id, num_tiles=None, indices=None, dtype='float32'):
    indices = range(num_tiles) if indices is None else indices
    return [np.array(PIL.Image.open(os.path.join(root, f'{image_id}_{i}.png')), dtype=dtype) for i in indices]


def load_test_tiles(root,
                    image_id,
                    img_size,
                    num_tiles=None,
                    resolution_idx=1,
                    pad_mode=0,
                    th=0.01):
    img = skimage.io.MultiImage(os.path.join(root, f'{image_id}.tiff'))[resolution_idx]
    tiles = get_tiles(img, img_size=img_size, num_tiles=num_tiles, pad_mode=pad_mode, th=th)
    return tiles


def stratify_batches(indices,
                     labels,
                     batch_size,
                     drop_last=False,
                     random_state=6969):
    """Returns a list of indices stratified by `labels` where each stratification is
    of size `bathc_size`
    """
    strat_indices = []

    num_batches = int(np.ceil(len(indices) / batch_size))
    remainder = len(indices) % batch_size
    num_batches = num_batches - 1 if remainder > 0 else num_batches

    if remainder > 0:
        remainder_indices = []
        remainder_labels = []
        rs = np.random.RandomState(5)
        last_idx = rs.choice(indices, size=remainder, replace=False)
        for idx in indices:
            if idx not in last_idx:
                remainder_indices.append(idx)
                remainder_labels.append(labels[idx])
    else:
        remainder_indices = indices
        remainder_labels = labels
        last_idx = []

    skf = StratifiedKFold(n_splits=num_batches,
                          shuffle=True,
                          random_state=random_state)

    for _, batch_idx in skf.split(remainder_indices, remainder_labels):
        strat_indices.extend(batch_idx)

    if not drop_last:
        strat_indices.extend(last_idx)

    return strat_indices


def load_img_stats(root, fold_id, filename='img_stats.json'):
    filepath = os.path.join(root, f'fold_{fold_id}', filename)
    print(f'Loading img stats for fold {fold_id} from {filepath}')
    with open(filepath, 'rb') as f:
        img_stats = eval(json.load(f))
    return img_stats
