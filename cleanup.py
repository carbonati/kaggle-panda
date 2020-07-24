import os
import shutil
import glob
import shutil
import argparse


def cleanup_log_dir(root, min_steps=5, keep_n=5):
    for model_name in os.listdir(root):
        model_dir = os.path.join(root, model_name)
        cleanup_ckpts(model_dir, min_steps, keep_n)


def cleanup_ckpts(model_dir, min_steps=5, keep_n=5):
    if not os.path.exists(os.path.join(model_dir, 'train_history.log')):
        return
    fold_dirs = glob.glob(os.path.join(model_dir, 'fold_*'))
    if len(fold_dirs) > 0:
        for fold_dir in fold_dirs:
            ckpt_files = []
            max_step = 0
            for fn in os.listdir(fold_dir):
                if fn.startswith('ckpt_'):
                    step = int(fn.split('_')[1])
                    if step > max_step:
                        max_step = step
                    ckpt_files.append(fn)
            if max_step < min_steps:
                print(f'removing fold directory {fold_dir}')
                shutil.rmtree(fold_dir)
            else:
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1]))
                if len(ckpt_files) > keep_n:
                    for fn in ckpt_files[:-keep_n]:
                        print(f'removing checkpoint {os.path.join(fold_dir, fn)}')
                        step = int(fn.split('_')[1])
                        os.remove(os.path.join(fold_dir, fn))
                        if os.path.exists(os.path.join(fold_dir, f'coef_{step:04d}.npy')):
                            os.remove(os.path.join(fold_dir, f'coef_{step:04d}.npy'))
        if not any([os.path.exists(f) for f in fold_dirs]):
            print(f'removing model directory {model_dir}')
            shutil.rmtree(model_dir)
    else:
        print(f'removing model directory {model_dir}')
        shutil.rmtree(model_dir)


def main(args):
    exp_dirs = glob.glob(os.path.join(args.root, f'*_exp*'))
    keep_n = args.keep_n
    min_steps = args.min_steps
    for exp_dir in exp_dirs:
        cleanup_log_dir(exp_dir, min_steps=min_steps, keep_n=keep_n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        '-r',
                        default='/root/workspace/kaggle-panda/models',
                        type=str,
                        help='Path to model directory.')
    parser.add_argument('--keep_n', '-n', default=5, type=int)
    parser.add_argument('--min_steps', '-m', default=5, type=int)
    args = parser.parse_args()
    main(args)
