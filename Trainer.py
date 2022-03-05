from random import sample
from HyperParameter import HyperParameters
from pickle import DICT
import torch
from torch.nn import functional as F
from typing import Dict
import numpy as np

from torch.functional import Tensor
from LoadAndSave import *

from EnvWrappers import MaskVelocityWrapper, PerturbationWrapper
from TrajectoryDataset import TrajectoryDataset
import time
from torch.utils.tensorboard import SummaryWriter

import pynvml
pynvml.nvmlInit()

class Trainer:
    def __init__(self,
                 env_name: str,
                 mask_velocity: bool,
                 experiment_name: str,
                 hp: HyperParameters,
                 asynchronous_environment: bool = False,
                 force_cpu_gather: bool = True,
                 checkpoint_frequency: int = 20,
                 workspace_path: str = './workspace') -> None:
        
        self.hp = hp
        self.env_name = env_name
        self.mask_velocity = mask_velocity
        self.obsv_dim, self.action_dim, self.continuous_action_space = get_env_space(env_name)
        self.base_checkpoint_path = f'{workspace_path}/checkpoints/{experiment_name}/'
        self.checkpoint_frequency = checkpoint_frequency
        
        self.train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda:0" if torch.cuda.is_available() and not force_cpu_gather else "cpu"
        self.min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
        self.asynchronous_environment = asynchronous_environment
        self.start_or_resume_from_checkpoint()

        self.best_reward = -1e6
        self.fail_to_improve_count = 0  # 没有提升的训练次数

        # 矢量化环境和标准gym环境一大不同就是会自动reset, 所以子环境的done生效时对应的state是reset后的初始状态
        # 子环境的step是并行还是串行, 因为电脑跑不起来很多线程所以区别不大
        self.env = gym.vector.make(self.env_name, self.hp.parallel_rollouts, asynchronous=self.asynchronous_environment)
        if self.mask_velocity:
            self.env = MaskVelocityWrapper(self.env)
        self.env = PerturbationWrapper(self.env, hp.noise)

        self.writer = SummaryWriter(log_dir=f"{workspace_path}/logs/{experiment_name}")
        self.SAVE_METRICS_TENSORBOARD = True

        RANDOM_SEED = 0
        torch.random.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        # torch.set_num_threads(8)  # 如果不是公用资源的话一般不需要指定运行线程数


    # 通过checkpoint恢复或初始化网络参数
    def start_or_resume_from_checkpoint(self):
        max_checkpoint_iteration = get_last_checkpoint_iteration(self.base_checkpoint_path)
        
        # 初始化
        if max_checkpoint_iteration == 0:
            self.actor = Actor(self.obsv_dim,
                        self.action_dim,
                        continuous_action_space=self.continuous_action_space,
                        hp = self.hp)
            self.critic = Critic(self.obsv_dim, self.hp)
            
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.hp.actor_learning_rate)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.hp.critic_learning_rate)
         
        # 从checkpoint恢复
        if max_checkpoint_iteration > 0:
            self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, hp, env_name, env_mask_velocity = load_from_checkpoint(self.base_checkpoint_path, max_checkpoint_iteration, 'cpu')
            
            assert env_name == self.env_name, "To resume training environment must match current settings."
            assert env_mask_velocity == self.mask_velocity, "To resume training model architecture must match current settings."
            # TODO, 因为patience, 所以暂时关闭
            # assert self.hp == hp, "To resume training hyperparameters must match current settings."
            for state in self.actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)

            for state in self.critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)
        self.iteration = max_checkpoint_iteration
    
    
    # 计算累计折扣回报
    def calc_discounted_return(self, rewards, discount, final_value):
        seq_len = len(rewards)
        discounted_returns = torch.zeros(seq_len)
        discounted_returns[-1] = rewards[-1] + discount * final_value
        for i in range(seq_len - 2, -1 , -1):
            discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
        return discounted_returns


    # 计算GAE
    def compute_advantages(self, rewards, values, discount, gae_lambda):
        deltas = rewards + discount * values[1:] - values[:-1]
        seq_len = len(rewards)
        advs = torch.zeros(seq_len + 1)     # 最后额外填充一个A值, 便于规范公式 
        multiplier = discount * gae_lambda
        for i in range(seq_len - 1, -1, -1):
            advs[i] = advs[i + 1] * multiplier  + deltas[i]
        return advs[:-1]


    # 矢量化环境执行step, 获取所有子环境的traj
    def gather_trajectories(self) ->  Dict[str, torch.Tensor]:
        
        # Initialise variables.
        obsv = self.env.reset()
        trajectory_data = {"states": [],  # shape = [rollout_steps, parallel_rollouts] 或 [rollout_steps, parallel_rollouts, dim]
                    "actions": [],
                    "action_probabilities": [],
                    "rewards": [],
                    "true_rewards": [],
                    "values": [],
                    "terminals": [],
                    "actor_hidden_states": [],
                    "actor_cell_states": [],
                    "critic_hidden_states": [],
                    "critic_cell_states": []}
        terminal = torch.ones(self.hp.parallel_rollouts)    # 初始状态时hidden_cell设为0

        with torch.no_grad():
            # 重置actor和critic的hidden_cell
            self.actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            # Take 1 additional step in order to collect the state and value for the final state.
            for _ in range(self.hp.rollout_steps):
                trajectory_data["actor_hidden_states"].append(self.actor.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["actor_cell_states"].append(self.actor.hidden_cell[1].squeeze(0).cpu())
                trajectory_data["critic_hidden_states"].append(self.critic.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["critic_cell_states"].append(self.critic.hidden_cell[1].squeeze(0).cpu())
                
                # 选择action
                state = torch.tensor(obsv, dtype=torch.float32)
                trajectory_data["states"].append(state)
                value = self.critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                trajectory_data["values"].append(value.squeeze(1).cpu())
                action_dist = self.actor(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                action = action_dist.sample().reshape(self.hp.parallel_rollouts, -1)
                if not self.actor.continuous_action_space:
                    action = action.squeeze(1)
                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

                # 执行step
                action_np = action.cpu().numpy()
                obsv, reward, done, _ = self.env.step(action_np)
                terminal = torch.tensor(done).float()
                transformed_reward = self.hp.scale_reward * torch.max(self.min_reward_values, torch.tensor(reward).float())
                                                                
                trajectory_data["rewards"].append(transformed_reward)
                trajectory_data["true_rewards"].append(torch.tensor(reward).float())
                trajectory_data["terminals"].append(terminal)
        
            # traj的最后额外计算一个v值, 便于后续gae之类的计算
            state = torch.tensor(obsv, dtype=torch.float32)
            value = self.critic(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
            trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

        # stack把list的子环境数据整合成大的tensor
        trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
        return trajectory_tensors  # shape = [rollout_steps, parallel_rollouts, dim]


    # 将所有子环境的traj根据episode切割, 并整合到一起
    def split_trajectories_episodes(self, trajectory_tensors: Dict[str, torch.Tensor]):

        len_episodes = []
        trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
        for i in range(self.hp.parallel_rollouts):  # 对每一个worker产生的traj切割
            terminals_tmp = trajectory_tensors["terminals"].clone()
            terminals_tmp[0, i] = 1     # 每一段traj的第一个位置和最后一个位置设为terminal
            terminals_tmp[-1, i] = 1
            split_points = (terminals_tmp[:, i] == 1).nonzero() + 1 # 找到切割点

            split_lens = split_points[1:] - split_points[:-1]
            split_lens[0] += 1
            
            len_episode = [split_len.item() for split_len in split_lens]
            len_episodes += len_episode
            for key, value in trajectory_tensors.items():
                # traj额外填充的V值算在最后一个episode里
                if key == "values":
                    value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))  # 最后一个episode额外补上一个value
                    # 前面的完整的episode最后填充一个 V_value==0 表示没有未来奖励
                    for j in range(len(value_split) - 1):
                        value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                    trajectory_episodes[key] += value_split
                else:
                    trajectory_episodes[key] += torch.split(value[:, i], len_episode)
        return trajectory_episodes, len_episodes    # shape = [episode个数, episode长度, dim], [episode个数]里面存的每个episode的长度


    # 把长度小于rollout_steps的episode都用0扩充到规定长度, 然后计算每个episode的A值和G值
    def pad_and_compute_returns(self, trajectory_episodes, len_episodes):

        episode_count = len(len_episodes)
        padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
        padded_trajectories["advantages"] = []
        padded_trajectories["discounted_returns"] = []

        for i in range(episode_count):
            single_padding = torch.zeros(self.hp.rollout_steps - len_episodes[i])
            for key, value in trajectory_episodes.items():
                if value[i].ndim > 1:
                    padding = torch.zeros(self.hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
                else:
                    padding = torch.zeros(self.hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
                padded_trajectories[key].append(torch.cat((value[i], padding)))
            padded_trajectories["advantages"].append(torch.cat((self.compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                            values=trajectory_episodes["values"][i],
                                                            discount=self.hp.discount,
                                                            gae_lambda=self.hp.gae_lambda), single_padding)))
            padded_trajectories["discounted_returns"].append(torch.cat((self.calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                        discount=self.hp.discount,
                                                                        final_value=trajectory_episodes["values"][i][-1]), single_padding)))
        return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()} 
        return_val["seq_len"] = torch.tensor(len_episodes)

        return return_val   # shape = [episode个数, rollout_steps, dim]
 

    def train(self):
        print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
        while self.iteration < self.hp.max_iterations:      

            self.actor = self.actor.to(self.gather_device)
            self.critic = self.critic.to(self.gather_device)
            start_gather_time = time.time()

            # traj的收集、裁剪并填充, 最后整理成episode
            trajectory_tensors = self.gather_trajectories()
            trajectory_episodes, len_episodes = self.split_trajectories_episodes(trajectory_tensors)
            trajectories = self.pad_and_compute_returns(trajectory_episodes, len_episodes)

            # 统计一共有多少明确terminal的episode
            complete_episode_count = trajectories["terminals"].sum().item()
            # 只统计有明确terminal标识的episode的reward总和
            terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
            # 计算episode的mean reward
            mean_reward =  terminal_episodes_rewards / complete_episode_count

            # 模型收敛停止的条件
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.fail_to_improve_count = 0
            else:
                self.fail_to_improve_count += 1
            
            if self.fail_to_improve_count > self.hp.patience:
                print(f"Policy has not yielded higher reward for {self.hp.patience} iterations...  Stopping now.")
                break


            alpha, belta = min((self.iteration / 20000)**2, 1.0), 0.6
            start_iteration = 8000
            total_episode_count = len(trajectories["terminals"])
            total_episode_idx = None  # minibatch中所有episode根据指标进行idx的由小到大排序
            if 1 <= self.hp.sample <= 3 and self.iteration > start_iteration:
                reward_sum = trajectories["rewards"].sum(axis=1)
                reward_mean = reward_sum / trajectories["seq_len"]
                total_episode_idx = reward_mean.argsort()
            elif 4 <= self.hp.sample <= 6:
                # 计算deltas相关
                deltas = trajectories["rewards"] + self.hp.discount * trajectories["values"][:, 1:] - trajectories["values"][:, :-1]
                deltas_abs_sum = torch.maximum(deltas, -deltas).sum(axis=1)
                deltas_abs_mean = deltas_abs_sum / trajectories["seq_len"]
                total_episode_idx = deltas_abs_mean.argsort()
                
            # 针对on policy的优先回放
            eps_idx = None
            if (self.hp.sample == 1 or self.hp.sample == 4) and self.iteration > start_iteration:
                eps_idx = total_episode_idx[:int(total_episode_count*belta)]

            elif self.hp.sample == 2 or self.hp.sample == 5:
                print("alpha: ", alpha, "belta: ", belta)
                sort_num = int(total_episode_count * alpha)
                random_num = total_episode_count - sort_num
                # 从排序好的episode中随机选出random_num个, 并将其从total_episode_idx中去除
                eps_idx_1 = np.random.choice(total_episode_idx, size=random_num, replace=False)
                eps_idx_2 = np.setdiff1d(total_episode_idx, eps_idx_1)[:int(sort_num*belta)]
                eps_idx = np.concatenate((eps_idx_1, eps_idx_2))

            elif self.hp.sample == 3 or self.hp.sample == 6:
                print("alpha: ", alpha, "belta: ", belta)
                sort_num = int(total_episode_count * alpha)
                random_num = total_episode_count - sort_num
                # 从排序好的episode中随机选出random_num个, 并将其从total_episode_idx中去除
                eps_idx_1 = np.random.choice(total_episode_idx, size=random_num, replace=False)
                total_episode_idx = np.setdiff1d(total_episode_idx, eps_idx_1)
                # eps_idx_2 = np.setdiff1d(total_episode_idx, eps_idx_1)[:int(sort_num*belta)]
                rank_prob = np.array([1/i for i in range(1, len(total_episode_idx)+1)])
                sum_prob = sum(rank_prob)
                rank_prob = rank_prob / sum_prob
                eps_idx_2 = np.random.choice(total_episode_idx, size=int(sort_num*belta), replace=False, p=rank_prob)
                eps_idx = np.concatenate((eps_idx_1, eps_idx_2))

            # 对于采样出的episode进行提取
            if eps_idx is not None:
                print("before sample: ", trajectories["states"].size())
                for key, value in trajectories.items():
                    trajectories[key] = value[eps_idx]
                print("after  sample: ", trajectories["states"].size())

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("Before Traj: ", info.used / (1024 * 1024 * 1024))

            trajectory_dataset = TrajectoryDataset(trajectories, batch_size=self.hp.batch_size,
                                            device=self.train_device, batch_len=self.hp.recurrent_seq_len, rollout_steps=self.hp.rollout_steps)
            end_gather_time = time.time()
            start_train_time = time.time()

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("After Traj : ", info.used / (1024 * 1024 * 1024))
            
            self.actor = self.actor.to(self.train_device)
            self.critic = self.critic.to(self.train_device)

            # PPO算法开始训练
            for epoch_idx in range(self.hp.ppo_epochs):
                # 详见__next__方法
                for batch in trajectory_dataset:   
                    # 因为选取的序列是连续的, 所以取序列最开头的hidden_cell参与网络
                    self.actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                    
                    self.actor_optimizer.zero_grad()
                    action_dist = self.actor(batch.states)
                    action_probabilities = action_dist.log_prob(batch.actions[-1, :].to(self.train_device)).to(self.train_device)
                    probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                    surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                    surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - self.hp.ppo_clip, 1. + self.hp.ppo_clip) * batch.advantages[-1, :]
                    surrogate_loss_2 = action_dist.entropy().to(self.train_device)
                    actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(self.hp.entropy_factor * surrogate_loss_2)
                    actor_loss.backward() 
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)  # 如果所有参数的gradient组成的向量的L2范数大于max norm, 那么需要根据L2范数/max norm进行缩放
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    self.critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                    values = self.critic(batch.states)
                    critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                    critic_loss.backward() 
                    self.critic_optimizer.step()

            torch.cuda.empty_cache()    # 定期清理GPU缓存

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("After clear: ", info.used / (1024 * 1024 * 1024))
            end_train_time = time.time()
            print(f"Iteration: {self.iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
                f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
                f"Train time: {end_train_time - start_train_time:.2f}s")

            if self.SAVE_METRICS_TENSORBOARD:
                self.writer.add_scalar("complete_episode_count", complete_episode_count, self.iteration)
                self.writer.add_scalar("total_reward", mean_reward , self.iteration)
                self.writer.add_scalar("actor_loss", actor_loss, self.iteration)
                self.writer.add_scalar("critic_loss", critic_loss, self.iteration)
                self.writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), self.iteration)
            if self.iteration % self.checkpoint_frequency == 0:
                save_checkpoint(self.base_checkpoint_path, self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, self.iteration, self.hp, self.env_name, self.mask_velocity)
            self.iteration += 1
            
        return self.best_reward 