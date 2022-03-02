from platform import platform
from dataclasses import dataclass


# Default Hyperparameters
SCALE_REWARD:         float = 0.01
MIN_REWARD:           float = -1000.
HIDDEN_SIZE:          float = 128
BATCH_SIZE:           int   = 512
DISCOUNT:             float = 0.99
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2
PPO_EPOCHS:           int   = 10
MAX_GRAD_NORM:        float = 1.
ENTROPY_FACTOR:       float = 0.
ACTOR_LEARNING_RATE:  float = 1e-4
CRITIC_LEARNING_RATE: float = 1e-4
RECURRENT_SEQ_LEN:    int = 8
RECURRENT_LAYERS:     int = 1    
ROLLOUT_STEPS:        int = 2048
PARALLEL_ROLLOUTS:    int = 8
PATIENCE:             int = 200
TRAINABLE_STD_DEV:    bool = False 
INIT_LOG_STD_DEV:     float = 0.0



# 修饰数据类, 用于方便整理属性, 避免了init方法等等
@dataclass
class HyperParameters():
    scale_reward:         float = SCALE_REWARD              # reward的尺度缩放量
    min_reward:           float = MIN_REWARD                # 限制最小reward的大小
    hidden_size:          float = HIDDEN_SIZE               # LSTM网络输出维度
    batch_size:           int   = BATCH_SIZE                # 数据批量
    discount:             float = DISCOUNT                  # 折扣因子
    gae_lambda:           float = GAE_LAMBDA                # GAE lambda系数
    ppo_clip:             float = PPO_CLIP                  # PPO裁剪系数
    ppo_epochs:           int   = PPO_EPOCHS                # batch的重复使用次数
    max_grad_norm:        float = MAX_GRAD_NORM             # 梯度裁剪
    entropy_factor:       float = ENTROPY_FACTOR            # 熵损失的系数
    actor_learning_rate:  float = ACTOR_LEARNING_RATE       # actor学习率       
    critic_learning_rate: float = CRITIC_LEARNING_RATE      # critic学习率
    recurrent_seq_len:    int = RECURRENT_SEQ_LEN           # 指定参与训练的seq长度
    recurrent_layers:     int = RECURRENT_LAYERS            # LSTM隐层的层数
    rollout_steps:        int = ROLLOUT_STEPS               # episode最大长度
    parallel_rollouts:    int = PARALLEL_ROLLOUTS           # gym矢量化环境同时运行多个子环境
    patience:             int = PATIENCE                    # 达到一定次数未突破, 结束训练
    # LSTM
    use_lstm:             bool = True                       # 网络模型是否加入LSTM                       
    # 连续动作空间中使用
    trainable_std_dev:    bool = TRAINABLE_STD_DEV          # 标准差是否需要梯度
    init_log_std_dev:     float = INIT_LOG_STD_DEV          # 初始标准差的log值
    # 终止条件
    max_iterations:       int = 1000000                     # 最大训练迭代次数
    noise:                float = 0.0                       # 对state施加标准正态分布噪声扰动的尺度缩放量