import torch
import numpy as np
import math
from TrajectorBatch import TrajectorBatch


# 设计一个方便从trajectories收集 训练用的batch 的dataset
class TrajectoryDataset():
    def __init__(self, trajectories, batch_size, device, batch_len, rollout_steps):
        
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len 
        # 保证__next__函数中np.linspace收集的batch_len个数据来自同一个episode
        # 不够batch_len长度的episode由于truncated_seq_len == 0, cumsum过程中不会记录该episode起始索引, 所以不会在np.digitize中被选出
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, rollout_steps)
        # 通过累计和的计算来实现得到每一个episode起始索引的目的
        self.cumsum_seq_len =  np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size
        
    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self
        
    def __next__(self):
        # 基本上只使用了(1/batch_len)的有效数据, 不过ppo_epochs可以重复使用, 把数据利用率拾回来
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size) 
            # 随机选batch_size个索引
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            # 从可选索引中将已选索引删除
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            # 通过起始索引的递增数列来找出start_idx属于哪些episode
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1
            # 确定start_idx在自己的episode的哪个位置
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            # 根据seq_idx收集连续的batch_len个数据, 这些索引将参与到训练的 批batch
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            # 例子: a = np.array([[[1],[2],[3]], [[4],[5],[6]],[[7],[8],[9]]])
            # print(a.shape)
            # c = a[[0, 1], [[1, 1], [2, 2], [0, 0]]]
            # print(c)
            # print(c.shape)

            # eps_idx.shape = (batch_size,), series_idx.shape = (batch_len, batch_size), trajectories.shape = (episode_num, rollouts_steps, dim)
            # 以eps_idx的batch_size_value做axis=0, series_idx的batch_size_value做axis=1, 这样可以在trajectories找到batch_size个数据, 
            # 对应哪一个episode第几步, 然后再和batch_len结合, 最终batch.shape = [batch_len, batch_size, dim]
            return TrajectorBatch(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)

