import os
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.detach()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
        
def get_last_checkpoint(save_dir, name):
    """ Find the last checkpoint available in the experiment folder """
    ckpt_dir = os.path.join(save_dir, name, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, None

    restart_ckpt = os.path.join(ckpt_dir, 'ckpt_restart.pt')
    if os.path.isfile(restart_ckpt):
        return torch.load(restart_ckpt), 'ckpt_restart.pt'
    else:
        ckpts = sorted([c for c in os.listdir(ckpt_dir) if 'ckpt_' in c],
                       key=lambda s: int(s.split('ckpt_')[1].split('.')[0]))
        if len(ckpts) > 0:
            return torch.load(os.path.join(ckpt_dir, ckpts[-1])), ckpts[-1]
    return None, None

def generate_masks(seq_len, list_perc_mask=[0, 10, 25], chunk_size=[1, 1], n=1000):
    """
    Generate masks for pretraining using the BERT paradigm
    Args:
        - perc_mask: corresponds to the percentage of 0 in the mask in avg to add
        - seq_len: length of the sequence
    Return:
        - mask: np.array of shape [T] whose values are either 0 or 1
    """
    assert len(chunk_size) == 2
    _min = chunk_size[0]
    if seq_len < chunk_size[0]:
        _min = 1
    range_chunk_size = range(_min, min([seq_len, chunk_size[1]]) + 1, 1)

    list_mask = []
    _n = (n // len(list_perc_mask)) + 1
    for perc_mask in list_perc_mask:
        for _ in range(_n):
            mask = np.ones(seq_len)
            while mask.mean() > 1 - (perc_mask / 100.):
                size = np.random.choice(range_chunk_size)
                start = np.random.choice(range(seq_len - size + 1))
                for i in range(start, start + size):
                    mask[i] = 0
            list_mask.append(mask)

    mask = np.stack(list_mask)
    np.random.shuffle(mask)
    return mask[:n]