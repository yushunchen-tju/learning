import os.path

import numpy as np
import pandas as pd
from datetime import timedelta
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader
# from utils.timefeatures import time_features


class OceanDataset(Dataset):
    """
    读取你们海洋 npy 数据，返回 (input, output)
    input : [pre_seq_length, C, H, W]
    output: [aft_seq_length, C, H, W]
    """

    def __init__(self, root, is_train=True, data_name='ocean',
                 startDate='20000101', endDate='20000105',
                 freq='12H', horizon=3,
                 pre_seq_length=7, aft_seq_length=7,
                 merge_depth_to_channel=True):
        super().__init__()
        self.npy_path = os.path.join(root,'features_ocean_norm')
        self.is_train = is_train
        self.freq = freq
        self.horizon = horizon
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.merge_depth_to_channel = merge_depth_to_channel

        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
        # 保证不会越界
        self.length = len(self.keys) - horizon - aft_seq_length - 1

        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def nctonumpy(self, dataset_path):
        return np.load(dataset_path, allow_pickle=True)

    # def generate_time_features(self, dates):
    #     mask = []
    #     for time_in in dates:
    #         time_mask = time_features(time_in, freq='D')
    #         mask.append(time_mask)
    #     mask = np.array(mask).squeeze()
    #     return mask

    def LoadData(self, key):
        end_time = key + timedelta(days=self.horizon)

        seq_x_keys = [key + timedelta(days=i) for i in range(self.pre_seq_length)]
        seq_y_keys = [end_time + timedelta(days=i) for i in range(self.aft_seq_length)]

        def build_path(time_key):
            time_str = time_key.strftime('%Y%m%d%H')
            parts = self.npy_path.split("/")
            target_substr = parts[-2]
            return f"{self.npy_path}/ocean_norm_{time_str[0:8]}.npy"

        seq_x_data = [self.nctonumpy(build_path(t)) for t in seq_x_keys]
        seq_y_data = [self.nctonumpy(build_path(t)) for t in seq_y_keys]

        seq_x_data = np.array(seq_x_data)  # (T, F, D, H, W)
        seq_y_data = np.array(seq_y_data)

        # seq_x_mark = self.generate_time_features(seq_x_keys)
        # seq_y_mark = self.generate_time_features(seq_y_keys)

        return seq_x_data, seq_y_data

    def __getitem__(self, index):
        key = self.keys[index]
        x, y= self.LoadData(key)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        # 如果不需要时间特征，下面两行可以删掉
        # x_mark = torch.from_numpy(x_mark).float()
        # y_mark = torch.from_numpy(y_mark).float()
        x = x[:,0:1,:,:,:]
        y = y[:,0:1,:,:,:]
        # 对齐 OpenSTL 的 [T, C, H, W]
        if x.dim() == 5 and self.merge_depth_to_channel:
            T, F, D, H, W = x.shape
            x = x.reshape(T, F * D, H, W)
            Tp, Fp, Dp, Hp, Wp = y.shape
            y = y.reshape(Tp, Fp * Dp, Hp, Wp)

        # 只返回 input/output（和 MNIST 一致）
        return x, y

        # 如果你要把时间特征也给模型，就改成：
        # return x, y, x_mark, y_mark

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers=4, **kwargs):

    train_start = kwargs['train_start']
    train_end = kwargs['train_end']
    val_start = kwargs['val_start']
    val_end = kwargs['val_end']
    test_start = kwargs['test_start']
    test_end = kwargs['test_end']
    pre_seq_length = kwargs['pre_seq_length']
    aft_seq_length = kwargs['aft_seq_length']
    freq = kwargs['freq']
    horizon = kwargs['horizon']
    distributed = kwargs['distributed']
    use_prefetcher = kwargs['use_prefetcher']
    drop_last = kwargs['drop_last']


    train_set = OceanDataset(
        root=data_root,
        startDate=train_start, endDate=train_end,
        freq=freq, horizon=horizon,
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length
    )

    val_set = OceanDataset(
        root=data_root,
        startDate=val_start, endDate=val_end,
        freq=freq, horizon=horizon,
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length
    )

    test_set = OceanDataset(
        root=data_root,
        startDate=test_start, endDate=test_end,
        freq=freq, horizon=horizon,
        pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length
    )

    dataloader_train = create_loader(
        train_set, batch_size=batch_size,
        shuffle=True, is_training=True,
        pin_memory=True, drop_last=True,
        num_workers=num_workers, persistent_workers=True,
        distributed=distributed, use_prefetcher=use_prefetcher
    )

    dataloader_vali = create_loader(
        val_set, batch_size=val_batch_size,
        shuffle=False, is_training=False,
        pin_memory=True, drop_last=drop_last,
        num_workers=num_workers, distributed=distributed,
        use_prefetcher=use_prefetcher
    )

    dataloader_test = create_loader(
        test_set, batch_size=val_batch_size,
        shuffle=False, is_training=False,
        pin_memory=True, drop_last=drop_last,
        num_workers=num_workers, distributed=distributed,
        use_prefetcher=use_prefetcher
    )

    return dataloader_train, dataloader_vali, dataloader_test
