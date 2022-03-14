from os import stat
from HyperParameter import HyperParameters
import torch
from torch.nn import functional as F
from typing import Dict
import numpy as np
from pettingzoo.mpe import simple_tag_v2

from MultiLoadAndSave import *

from EnvWrappers import MaskVelocityWrapper, PerturbationWrapper
from TrajectoryDataset import TrajectoryDataset
import time
from torch.utils.tensorboard import SummaryWriter

import pynvml
pynvml.nvmlInit()

class MultiTrainer:
    def __init__(self,
                 env_name: str,
                 experiment_name: str,
                 hp: HyperParameters,
                 force_cpu_gather: bool = True,
                 checkpoint_frequency: int = 20,
                 workspace_path: str = './workspace') -> None:
        
        self.hp = hp
        self.env_name = env_name
        self.global_obsv_dim, self.team1_obsv_dim, self.team2_obsv_dim, self.action_dim, self.continuous_action_space = get_env_space(env_name, self.hp)
        self.base_checkpoint_path = f'{workspace_path}/checkpoints/{experiment_name}/'
        self.checkpoint_frequency = checkpoint_frequency
        
        self.train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda:0" if torch.cuda.is_available() and not force_cpu_gather else "cpu"
        self.min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
        self.start_or_resume_from_checkpoint()

        self.team1_best_reward = -1e6
        self.team2_best_reward = -1e6
        self.fail_to_improve_count = 0  # 没有提升的训练次数

        self.env = simple_tag_v2.parallel_env(num_good=self.hp.num_team2, num_adversaries=self.hp.num_team1, \
            num_obstacles=self.hp.num_obstacles, max_cycles=self.hp.rollout_steps, continuous_actions=self.continuous_action_space)
        # self.env = PerturbationWrapper(self.env, hp.noise)

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
            self.team1_actor = Actor(self.team1_obsv_dim,
                        self.action_dim,
                        continuous_action_space=self.continuous_action_space,
                        hp = self.hp)
            self.team2_actor = Actor(self.team2_obsv_dim,
                        self.action_dim,
                        continuous_action_space=self.continuous_action_space,
                        hp = self.hp)
            # TODO, 换global
            self.team1_critic = Critic(self.team1_obsv_dim, self.hp)
            self.team2_critic = Critic(self.team2_obsv_dim, self.hp)

            self.team1_actor_optimizer = torch.optim.AdamW(self.team1_actor.parameters(), lr=self.hp.actor_learning_rate)
            self.team1_critic_optimizer = torch.optim.AdamW(self.team1_critic.parameters(), lr=self.hp.critic_learning_rate)
            self.team2_actor_optimizer = torch.optim.AdamW(self.team2_actor.parameters(), lr=self.hp.actor_learning_rate)
            self.team2_critic_optimizer = torch.optim.AdamW(self.team2_critic.parameters(), lr=self.hp.critic_learning_rate)

        # 从checkpoint恢复
        if max_checkpoint_iteration > 0:
            self.team1_actor, self.team1_critic, self.team1_actor_optimizer, self.team1_critic_optimizer, self.team2_actor, \
                self.team2_critic, self.team2_actor_optimizer, self.team2_critic_optimizer, hp, env_name = load_from_checkpoint(self.base_checkpoint_path, max_checkpoint_iteration, 'cpu')
            
            assert env_name == self.env_name, "To resume training environment must match current settings."
            # 因为patience, 所以有时候需要暂时关闭
            # assert self.hp == hp, "To resume training hyperparameters must match current settings."
            for state in self.team1_actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)

            for state in self.team1_critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)

            for state in self.team2_actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.train_device)

            for state in self.team2_critic_optimizer.state.values():
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
        team1_trajectory_episodes = {"states": [],  # shape = [rollout_steps, parallel_rollouts] 或 [rollout_steps, parallel_rollouts, dim]
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
        team2_trajectory_episodes = {"states": [],  # shape = [rollout_steps, parallel_rollouts] 或 [rollout_steps, parallel_rollouts, dim]
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
        trajectory_episodes = [dict()] * self.env.num_agents
        for i in range(self.env.num_agents):
            trajectory_episodes[i] = {"states": [],  # 每个agent对应一个traj_episodes, 最后拼接到team1和team2
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
        agents = [agent for agent in self.env.agents]
        terminal = torch.ones(self.hp.parallel_rollouts)    # 初始状态时hidden_cell设为0

        with torch.no_grad():
            for t in range(self.hp.num_episodes):  # 进行多少局
                # TODO, 换global修改关键词, torch.tensor(obsv[agents[i]])
                obsv = self.env.reset()
                global_obsv = [v for v in obsv.values()]
                global_obsv = torch.tensor(np.concatenate(tuple(global_obsv)), dtype=torch.float32)

                # 重置actor和critic的hidden_cell
                self.team1_actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
                self.team2_actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
                self.team1_critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)
                self.team2_critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)
                # MPE环境一定执行rollout_steps步, 无提前done, 所以一次for循环产生的traj一定是一个完整的episode, 后面也省略了split和pad
                for t in range(self.hp.rollout_steps):
                    # self.env.render()
                    # time.sleep(0.05)
                    for i in range(self.hp.num_team1):
                        trajectory_episodes[i]["actor_hidden_states"].append(self.team1_actor.hidden_cell[0].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["actor_cell_states"].append(self.team1_actor.hidden_cell[1].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["critic_hidden_states"].append(self.team1_critic.hidden_cell[0].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["critic_cell_states"].append(self.team1_critic.hidden_cell[1].squeeze(0).squeeze(0).cpu())
                    for i in range(self.hp.num_team1, self.hp.num_team1 + self.hp.num_team2):
                        trajectory_episodes[i]["actor_hidden_states"].append(self.team2_actor.hidden_cell[0].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["actor_cell_states"].append(self.team2_actor.hidden_cell[1].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["critic_hidden_states"].append(self.team2_critic.hidden_cell[0].squeeze(0).squeeze(0).cpu())
                        trajectory_episodes[i]["critic_cell_states"].append(self.team2_critic.hidden_cell[1].squeeze(0).squeeze(0).cpu())

                    # 选择action TODO, 看一下走了一步之后, self.env.state_space如何变化, 能不能直接拿来当作state
                    # critic使用全局obsv, actor使用局部obsv
                    actions = {}
                    for i in range(self.hp.num_team1):
                        a = torch.tensor(obsv[agents[i]])
                        value = self.team1_critic(torch.tensor(obsv[agents[i]]).unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                        trajectory_episodes[i]["values"].append(value.squeeze(1).cpu())
                        state = torch.tensor(obsv[agents[i]], dtype=torch.float32)
                        trajectory_episodes[i]["states"].append(state)
                        action_dist = self.team1_actor(state.unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                        action = action_dist.sample()
                        trajectory_episodes[i]["actions"].append(action.cpu())
                        trajectory_episodes[i]["action_probabilities"].append(action_dist.log_prob(action).cpu())
                        actions[agents[i]] = action.item()

                    for i in range(self.hp.num_team1, self.hp.num_team1 + self.hp.num_team2):
                        value = self.team2_critic(torch.tensor(obsv[agents[i]]).unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                        trajectory_episodes[i]["values"].append(value.squeeze(1).cpu())
                        state = torch.tensor(obsv[agents[i]], dtype=torch.float32)
                        trajectory_episodes[i]["states"].append(state)
                        action_dist = self.team2_actor(state.unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                        action = action_dist.sample()
                        trajectory_episodes[i]["actions"].append(action.cpu())
                        trajectory_episodes[i]["action_probabilities"].append(action_dist.log_prob(action).cpu())
                        actions[agents[i]] = action.item()

                    # 执行step
                    obsv, reward, done, _ = self.env.step(actions)
                    global_obsv = [v for v in obsv.values()]
                    global_obsv = torch.tensor(np.concatenate(tuple(global_obsv)), dtype=torch.float32)
                    # TODO, reward看一下, 官网说是累积, 看看到底是不是
                    for i in range(self.hp.num_team1 + self.hp.num_team2):
                        terminal = torch.tensor(done[agents[i]]).float()
                        true_reward = torch.tensor(reward[agents[i]]).float()   # 碰撞agent, 所有捕猎者+10, 单个agent-10; agent有额外的出界惩罚
                        transformed_reward = self.hp.scale_reward * torch.max(self.min_reward_values, true_reward)
                        trajectory_episodes[i]["rewards"].append(transformed_reward)
                        trajectory_episodes[i]["true_rewards"].append(true_reward)
                        trajectory_episodes[i]["terminals"].append(terminal)

                # traj的最后额外计算一个v值, 便于后续gae之类的计算
                for i in range(self.hp.num_team1):
                    value = self.team1_critic(torch.tensor(obsv[agents[i]]).unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                    trajectory_episodes[i]["values"].append(value.squeeze(1).cpu()) # 这里不想加(1 - done[agents[i]]), 因为done一定是true但是我想要value
                for i in range(self.hp.num_team1, self.hp.num_team1 + self.hp.num_team2):
                    value = self.team2_critic(torch.tensor(obsv[agents[i]]).unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                    trajectory_episodes[i]["values"].append(value.squeeze(1).cpu())
                
                # 把trajdata的数据整合进team1_traj 和 team2_traj并清空
                for i in range(self.hp.num_team1):
                    for key in team1_trajectory_episodes.keys():
                        team1_trajectory_episodes[key].append(torch.stack(trajectory_episodes[i][key]))
                        trajectory_episodes[i][key] = []
                for i in range(self.hp.num_team1, self.hp.num_team1 + self.hp.num_team2):
                    for key in team2_trajectory_episodes.keys():
                        team2_trajectory_episodes[key].append(torch.stack(trajectory_episodes[i][key]))
                        trajectory_episodes[i][key] = []
    
        return team1_trajectory_episodes, team2_trajectory_episodes  # shape = [episode个数, episode长度, dim]


    # 把长度小于rollout_steps的episode都用0扩充到规定长度, 然后计算每个episode的A值和G值
    def pad_and_compute_returns(self, team1_trajectory_episodes, team2_trajectory_episodes):

        team1_trajectory_episodes["advantages"] = []
        team1_trajectory_episodes["discounted_returns"] = []
        team2_trajectory_episodes["advantages"] = []
        team2_trajectory_episodes["discounted_returns"] = []

        team1_episode_count, team2_episode_count = len(team1_trajectory_episodes["states"]), len(team2_trajectory_episodes["states"]) 
        for i in range(team1_episode_count):
            team1_trajectory_episodes["advantages"].append(self.compute_advantages(rewards=team1_trajectory_episodes["rewards"][i],
                                                            values=team1_trajectory_episodes["values"][i],
                                                            discount=self.hp.discount,
                                                            gae_lambda=self.hp.gae_lambda))
            team1_trajectory_episodes["discounted_returns"].append(self.calc_discounted_return(rewards=team1_trajectory_episodes["rewards"][i],
                                                                        discount=self.hp.discount,
                                                                        final_value=team1_trajectory_episodes["values"][i][-1]))
        for i in range(team2_episode_count):
            team2_trajectory_episodes["advantages"].append(self.compute_advantages(rewards=team2_trajectory_episodes["rewards"][i],
                                                            values=team2_trajectory_episodes["values"][i],
                                                            discount=self.hp.discount,
                                                            gae_lambda=self.hp.gae_lambda))
            team2_trajectory_episodes["discounted_returns"].append(self.calc_discounted_return(rewards=team2_trajectory_episodes["rewards"][i],
                                                                        discount=self.hp.discount,
                                                                        final_value=team2_trajectory_episodes["values"][i][-1]))
        
        team1_trajectories = {k: torch.stack(v).squeeze(-1) for k, v in team1_trajectory_episodes.items()} 
        team2_trajectories = {k: torch.stack(v) for k, v in team2_trajectory_episodes.items()} 
        team1_trajectories["seq_len"] = torch.tensor([self.hp.rollout_steps] * team1_episode_count)
        team2_trajectories["seq_len"] = torch.tensor([self.hp.rollout_steps] * team2_episode_count)

        return team1_trajectories, team2_trajectories   # shape = [episode个数, rollout_steps, dim]
 

    def train(self):
        print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
        while self.iteration < self.hp.max_iterations:      

            self.team1_actor = self.team1_actor.to(self.gather_device)
            self.team2_actor = self.team2_actor.to(self.gather_device)
            self.team1_critic = self.team1_critic.to(self.gather_device)
            self.team2_critic = self.team2_critic.to(self.gather_device)
            start_gather_time = time.time()

            # traj的收集、裁剪并填充, 最后整理成episode
            team1_trajectory_episodes, team2_trajectory_episodes = self.gather_trajectories()
            team1_trajectories, team2_trajectories = self.pad_and_compute_returns(team1_trajectory_episodes, team2_trajectory_episodes)

            # 计算episode的mean reward
            team1_episode_count, team2_episode_count = len(team1_trajectory_episodes["states"]), len(team2_trajectory_episodes["states"]) 
            team1_mean_reward = team1_trajectories["true_rewards"].sum() / team1_episode_count
            team2_mean_reward = team2_trajectories["true_rewards"].sum() / team2_episode_count

            # 模型收敛停止的条件, 因为是对抗任务, 所以不管是adversory还是agent不增长都可以停止
            if team1_mean_reward > self.team1_best_reward:
                self.team1_best_reward = max(self.team1_best_reward, team1_mean_reward)
                self.fail_to_improve_count = 0
            elif team2_mean_reward > self.team2_best_reward:
                self.team2_best_reward = max(self.team2_best_reward, team2_mean_reward)
                self.fail_to_improve_count = 0
            else:
                self.fail_to_improve_count += 1
            
            if self.fail_to_improve_count > self.hp.patience:
                print(f"Policy has not yielded higher reward for {self.hp.patience} iterations...  Stopping now.")
                break


            # alpha, belta = min((self.iteration / 20000)**2, 1.0), 0.6
            # start_iteration = 8000
            # total_episode_count = len(team1_trajectories["terminals"])
            # total_episode_idx = None  # minibatch中所有episode根据指标进行idx的由小到大排序
            # if 1 <= self.hp.sample <= 3:
            #     reward_sum = team1_trajectories["rewards"].sum(axis=1)
            #     reward_mean = reward_sum / team1_trajectories["seq_len"]
            #     total_episode_idx = reward_mean.argsort()
            # elif 4 <= self.hp.sample <= 6:
            #     # 计算deltas相关
            #     deltas = team1_trajectories["rewards"] + self.hp.discount * team1_trajectories["values"][:, 1:] - team1_trajectories["values"][:, :-1]
            #     deltas_abs_sum = torch.maximum(deltas, -deltas).sum(axis=1)
            #     deltas_abs_mean = deltas_abs_sum / team1_trajectories["seq_len"]
            #     total_episode_idx = deltas_abs_mean.argsort()
                
            # # 针对on policy的优先回放
            # eps_idx = None
            # if (self.hp.sample == 1 or self.hp.sample == 4) and self.iteration > start_iteration:
            #     eps_idx = total_episode_idx[:int(total_episode_count*belta)]

            # elif self.hp.sample == 2 or self.hp.sample == 5:
            #     print("alpha: ", alpha, "belta: ", belta)
            #     sort_num = int(total_episode_count * alpha)
            #     random_num = total_episode_count - sort_num
            #     # 从排序好的episode中随机选出random_num个, 并将其从total_episode_idx中去除
            #     eps_idx_1 = np.random.choice(total_episode_idx, size=random_num, replace=False)
            #     eps_idx_2 = np.setdiff1d(total_episode_idx, eps_idx_1)[:int(sort_num*belta)]
            #     eps_idx = np.concatenate((eps_idx_1, eps_idx_2))

            # elif self.hp.sample == 3 or self.hp.sample == 6:
            #     print("alpha: ", alpha, "belta: ", belta)
            #     sort_num = int(total_episode_count * alpha)
            #     random_num = total_episode_count - sort_num
            #     # 从排序好的episode中随机选出random_num个, 并将其从total_episode_idx中去除
            #     eps_idx_1 = np.random.choice(total_episode_idx, size=random_num, replace=False)
            #     total_episode_idx = np.setdiff1d(total_episode_idx, eps_idx_1)
            #     # eps_idx_2 = np.setdiff1d(total_episode_idx, eps_idx_1)[:int(sort_num*belta)]
            #     rank_prob = np.array([1/i for i in range(1, len(total_episode_idx)+1)])
            #     sum_prob = sum(rank_prob)
            #     rank_prob = rank_prob / sum_prob
            #     eps_idx_2 = np.random.choice(total_episode_idx, size=int(sort_num*belta), replace=False, p=rank_prob)
            #     eps_idx = np.concatenate((eps_idx_1, eps_idx_2))

            # # 对于采样出的episode进行提取
            # if eps_idx is not None:
            #     print("before sample: ", team1_trajectories["states"].size())
            #     for key, value in team1_trajectories.items():
            #         team1_trajectories[key] = value[eps_idx]
            #     print("after  sample: ", team1_trajectories["states"].size())

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("Before Traj: ", info.used / (1024 * 1024 * 1024))

            
            team1_trajectory_dataset = TrajectoryDataset(team1_trajectories, batch_size=self.hp.batch_size,
                                            device=self.train_device, batch_len=self.hp.recurrent_seq_len, rollout_steps=self.hp.rollout_steps)
            team2_trajectory_dataset = TrajectoryDataset(team2_trajectories, batch_size=self.hp.batch_size,
                                            device=self.train_device, batch_len=self.hp.recurrent_seq_len, rollout_steps=self.hp.rollout_steps)
            end_gather_time = time.time()
            start_train_time = time.time()

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("After Traj : ", info.used / (1024 * 1024 * 1024))
            
            self.team1_actor = self.team1_actor.to(self.train_device)
            self.team2_actor = self.team2_actor.to(self.train_device)
            self.team1_critic = self.team1_critic.to(self.train_device)
            self.team2_critic = self.team2_critic.to(self.train_device)

            # MAPPO算法开始训练
            for epoch_idx in range(self.hp.ppo_epochs):
                # 详见__next__方法
                for batch in team1_trajectory_dataset:
                    # 因为选取的序列是连续的, 所以取序列最开头的hidden_cell参与网络
                    self.team1_actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                    
                    self.team1_actor_optimizer.zero_grad()
                    action_dist = self.team1_actor(batch.states)
                    action_probabilities = action_dist.log_prob(batch.actions[-1, :].to(self.train_device)).to(self.train_device)
                    probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                    surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                    surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - self.hp.ppo_clip, 1. + self.hp.ppo_clip) * batch.advantages[-1, :]
                    surrogate_loss_2 = action_dist.entropy().to(self.train_device)
                    actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(self.hp.entropy_factor * surrogate_loss_2)
                    actor_loss.backward() 
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.team1_actor.parameters(), self.hp.max_grad_norm)  # 如果所有参数的gradient组成的向量的L2范数大于max norm, 那么需要根据L2范数/max norm进行缩放
                    self.team1_actor_optimizer.step()

                    self.team1_critic_optimizer.zero_grad()
                    self.team1_critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                    values = self.team1_critic(batch.states)
                    critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.team1_critic.parameters(), self.hp.max_grad_norm)
                    critic_loss.backward() 
                    self.team1_critic_optimizer.step()
            print(f"Team1:  Mean reward: {team1_mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, episode_count: {team1_episode_count}")
            if self.SAVE_METRICS_TENSORBOARD:
                self.writer.add_scalar("team1_episode_count", team1_episode_count, self.iteration)
                self.writer.add_scalar("team1_mean_reward", team1_mean_reward , self.iteration)
                self.writer.add_scalar("team1_actor_loss", actor_loss, self.iteration)
                self.writer.add_scalar("team1_critic_loss", critic_loss, self.iteration)
                self.writer.add_scalar("team1_policy_entropy", torch.mean(surrogate_loss_2), self.iteration)

            for epoch_idx in range(self.hp.ppo_epochs):
                for batch in team2_trajectory_dataset:   
                    # 因为选取的序列是连续的, 所以取序列最开头的hidden_cell参与网络
                    self.team2_actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                    
                    self.team2_actor_optimizer.zero_grad()
                    action_dist = self.team2_actor(batch.states)
                    action_probabilities = action_dist.log_prob(batch.actions[-1, :].to(self.train_device)).to(self.train_device)
                    probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                    surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                    surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - self.hp.ppo_clip, 1. + self.hp.ppo_clip) * batch.advantages[-1, :]
                    surrogate_loss_2 = action_dist.entropy().to(self.train_device)
                    actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(self.hp.entropy_factor * surrogate_loss_2)
                    actor_loss.backward() 
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.team2_actor.parameters(), self.hp.max_grad_norm)  # 如果所有参数的gradient组成的向量的L2范数大于max norm, 那么需要根据L2范数/max norm进行缩放
                    self.team2_actor_optimizer.step()

                    self.team2_critic_optimizer.zero_grad()
                    self.team2_critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                    values = self.team2_critic(batch.states)
                    critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.team2_critic.parameters(), self.hp.max_grad_norm)
                    critic_loss.backward() 
                    self.team2_critic_optimizer.step()
            print(f"Team2:  Mean reward: {team2_mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, episode_count: {team2_episode_count}")
            if self.SAVE_METRICS_TENSORBOARD:
                self.writer.add_scalar("team2_episode_count", team2_episode_count, self.iteration)
                self.writer.add_scalar("team2_mean_reward", team2_mean_reward , self.iteration)
                self.writer.add_scalar("team2_actor_loss", actor_loss, self.iteration)
                self.writer.add_scalar("team2_critic_loss", critic_loss, self.iteration)
                self.writer.add_scalar("team2_policy_entropy", torch.mean(surrogate_loss_2), self.iteration)

            torch.cuda.empty_cache()    # 定期清理GPU缓存

            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("After clear: ", info.used / (1024 * 1024 * 1024))
            
            end_train_time = time.time()
            print(f"Iteration: {self.iteration},  Gather time: {end_gather_time - start_gather_time:.2f}s, Train time: {end_train_time - start_train_time:.2f}s")

            if self.iteration % self.checkpoint_frequency == 0:
                save_checkpoint(self.base_checkpoint_path, self.team1_actor, self.team1_critic, self.team1_actor_optimizer, self.team1_critic_optimizer, \
                    self.team2_actor, self.team2_critic, self.team2_actor_optimizer, self.team2_critic_optimizer, self.iteration, self.hp, self.env_name)
            self.iteration += 1
            
        return self.team1_best_reward, self.team2_best_reward 