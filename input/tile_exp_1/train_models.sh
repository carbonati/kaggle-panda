steps=40
keep_prob=1
batch_size=4
experiment_name="tile_exp_1"
sleep_sec=20
num_gpus=4

export CUDA_VISIBLE_DEVICES=0,2,1,3

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus --distributed --fp_16 --fold_ids 0
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus --distributed --fp_16 --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus --distributed --fp_16 --fold_ids 2
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus --distributed --fp_16 --fold_ids 3
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus --distributed --fp_16 --fold_ids 4
