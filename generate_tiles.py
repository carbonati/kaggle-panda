import os
import json
import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
import argparse
import skimage.io
import cv2
import warnings
warnings.filterwarnings('ignore')

from panda.utils.generic_utils import load_config_from_yaml, compute_num_tiles
from panda.data.preprocess import get_tiles, remove_pen_marks


def main(config):
    """CV fold generation."""
    params = config['tiles']
    img_size = params['img_size']
    num_tiles = params['num_tiles']
    resolution_idx = params.pop('resolution_idx')
    pad_mode = params.get('pad_mode', 0)

    session_filename = f"train_tiles_{img_size}_{num_tiles or 0}_{resolution_idx}_{pad_mode}"
    session_dir = os.path.join(config['input']['output'], session_filename)
    tiles_filepath = os.path.join(session_dir, 'tiles.zip')
    masks_filepath = os.path.join(session_dir, 'masks.zip')
    meta_dir = os.path.join(session_dir, 'metadata')

    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    image_dir = config['input']['images']
    mask_dir = config['input'].get('masks')
    tile_dir = os.path.join(session_dir, 'tiles')
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    image_ids = sorted([fn.split('.')[0] for fn in os.listdir(image_dir)])
    if config['input'].get('marker_ids'):
        with open(config['input']['marker_ids'], 'rb') as f:
            marker_ids = pickle.load(f)
    else:
        marker_ids = []
    if config['input'].get('blacklist'):
        with open(config['input']['blacklist'], 'rb') as f:
            blacklist = pickle.load(f)
        image_ids = list(set(image_ids).difference(blacklist))

    num_tiles_data = []
    print(f'Saving tile data to {session_dir}')
    # with ZipFile(tiles_filepath, 'w') as tiles_file, ZipFile(masks_filepath, 'w') as masks_file:
    for image_id in tqdm(image_ids, total=len(image_ids)):
        print(image_id)
        img = skimage.io.MultiImage(os.path.join(image_dir, f'{image_id}.tiff'))[resolution_idx]
        # remove pen marks from slides
        if image_id in marker_ids:
            img = remove_pen_marks(img)
        if mask_dir is not None:
            filepath = os.path.join(mask_dir, f'{image_id}_mask.tiff')
            if os.path.exists(filepath):
                mask = skimage.io.MultiImage(filepath)[resolution_idx]
            else:
                mask = np.zeros_like(img, dtype=np.uint8)
            tiles, masks, meta = get_tiles(img, mask, **params)
        else:
            tiles, meta = get_tiles(img, **params)

        for i in range(len(tiles)):
            # tile = cv2.imencode('.png', cv2.cvtColor(tiles[i], cv2.COLOR_RGB2BGR))[1]
            # tiles_file.writestr(f'{image_id}_{i}.png', tile)
            cv2.imwrite(os.path.join(tile_dir, f'{image_id}_{i}.png'), tiles[i])
            if mask_dir is not None:
                mask_tile = cv2.imencode('.png', masks[i][...,0])[1]
                masks_file.writestr(f'{image_id}_{i}.png', mask_tile)

        with open(os.path.join(meta_dir, f'{image_id}.json'), 'w') as f:
            json.dump(str(meta), f)

        num_tiles_data.append([image_id, len(tiles)])

    df_num_tiles = pd.DataFrame(num_tiles_data, columns=['image_id', 'num_tiles'])
    df_num_tiles.to_csv(os.path.join(session_dir, 'num_tiles.csv'), index=False)
    print(f'Saved tiles and metadata to {session_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='generate_tiles_config.yaml',
                        type=str,
                        help='Path to tile preprocessing configuration file.')
    args = parser.parse_args()

    config = load_config_from_yaml(args.config_filepath)
    main(config)
