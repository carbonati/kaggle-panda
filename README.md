# kaggle-panda

Example machine learning pipeline for the [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment)

## Data Preparation

##### 1. Generate CV folds

Update `input/generate_cv_folds_config.yaml` then run

` python generate_cv_folds.py`

##### 2. Generate tiles

Update `input/generate_tiles_config.yaml` then run

` python generate_tiles.py`

##### 2. Generate image statistics

Update `input/generate_cv_folds_config.yaml` with the `session_dir` generated from step 1 and the tiles directory from step 2

` python generate_tiles.py`

## Training

update configurable files in `input/tile_exp_1` for a training session or `input/blend_exp_1` for blending experiments then run

`./input/blend_exp_1/train_models.sh`
