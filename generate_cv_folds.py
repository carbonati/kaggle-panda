import os
import json
import pickle
import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

from panda.utils.generic_utils import load_config_from_yaml
from panda.data.preprocess import generate_img_stats
from panda.utils import data_utils


def main(config):
    """CV fold generation."""
    df_panda = data_utils.load_data(config)

    if config['cv_folds'].get('session_name'):
        session_dir = os.path.join(config['input']['output'], config['cv_folds']['session_name'])
        cv_folds = data_utils.load_cv_folds(os.path.join(session_dir, 'cv_folds.p'))
        print(f'Loaded cv folds from {session_dir}')
    else:
        session_dt = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        if config.get('tag'):
            session_fn = f"cv_folds_{config['tag']}_{session_dt}"
        else:
            session_fn = f"cv_folds_{session_dt}"
        session_dir = os.path.join(config['input']['output'], session_fn)
        cv_folds = data_utils.generate_cv_folds(df_panda,
                                                **config['cv_folds'])

        if not os.path.exists(session_dir):
            print(f'Generating cv session directory @ {session_dir}')
            os.makedirs(session_dir)

        with open(os.path.join(session_dir, f'cv_folds.p'), 'wb') as f:
            pickle.dump(cv_folds, f)
        with open(os.path.join(session_dir, f'config.json'), 'w') as f:
            json.dump(config, f)
        print(f'Saved cv folds & config to {session_dir}')

    if config.get('img_stats'):
        img_version = config['img_stats']['root'].split('/')[-1]
        version_dir = os.path.join(session_dir, img_version)
        if not os.path.exists(version_dir):
            print(f'Generating image version directory @ {version_dir}')
            os.makedirs(version_dir)
        if not config['img_stats'].get('data'):
            args = img_version.split('_')
            config['img_stats'].update({
               'img_size': int(args[2]),
               'num_tiles': int(args[3]),
               'resolution_idx': int(args[4]),
               'pad_mode': int(args[5])
            })

        df_panda['fold'] = data_utils.get_fold_col(df_panda, cv_folds) # index_col ?
        fold_ids = list(set(df_panda['fold'].tolist()))
        params = config.get('img_stats')
        stratify = params.pop('stratify', None)
        # apped the tiles directory if not already
        if not params['root'].endswith('tiles'):
            params['root'] = os.path.join(params['root'], 'tiles')

        for fold_id in fold_ids:
            df_train = df_panda.loc[(df_panda['fold'] != fold_id) &
                                   (df_panda['fold'] != 'test')]
            output_dir = os.path.join(version_dir, f'fold_{fold_id}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if stratify is not None:
                img_stats = {}
                for col in stratify:
                    values = df_train[col].unique()
                    for v in values:
                        img_stats[v] = generate_img_stats(
                            df_train.loc[df_train[col] == v],
                            **params,
                        )
            else:
                img_stats = generate_img_stats(df_train,
                                               **params)

            with open(os.path.join(output_dir, 'img_stats.json'), 'w') as f:
                json.dump(str(img_stats), f)
            print(f'Saved to fold {fold_id} img stats to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='generate_cv_folds_config.yaml',
                        type=str,
                        help='Path to cv generation configuration file.')
    args = parser.parse_args()

    config = load_config_from_yaml(args.config_filepath)
    main(config)
