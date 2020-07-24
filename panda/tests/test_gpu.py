import torch

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        try:
            torch.cuda.set_device(gpu_id)
            torch.randn(1, device='cuda')
            print('GPU {} is good'.format(gpu_id))
        except Exception as exec:
            print('GPU {} is bad: {}'.format(gpu_id, exec))
