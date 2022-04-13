from HyperParameter import HyperParameters
import torch
import numpy as np
import time
from pettingzoo.mpe import simple_tag_v2, simple_world_comm_v2

from MultiLoadAndSave import *

from EnvWrappers import PerturbationWrapper


class MultiTester:
    def __init__(self,
                 env_name: str,
                 experiment_name: str,
                 force_cpu_gather: bool = True,
                 workspace_path: str = './workspace') -> None:
        
        self.env_name = env_name
        self.base_checkpoint_path = f'{workspace_path}/checkpoints/{experiment_name}/'
        self.start_or_resume_from_checkpoint()
        self.global_obsv_dim, self.team1_obsv_dim, self.team2_obsv_dim, self.action_dim, self.continuous_action_space = get_env_space(env_name, self.hp)

        self.train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda:0" if torch.cuda.is_available() and not force_cpu_gather else "cpu"

        # self.env = simple_tag_v2.parallel_env(num_good=self.hp.num_team2, num_adversaries=self.hp.num_team1, \
        #     num_obstacles=self.hp.num_obstacles, max_cycles=self.hp.rollout_steps, continuous_actions=self.continuous_action_space)
        self.env = simple_world_comm_v2.parallel_env(num_good=self.hp.num_team2, num_adversaries=self.hp.num_team1, num_obstacles=self.hp.num_obstacles, \
            num_forests=self.hp.num_forests, num_food=self.hp.num_food, max_cycles=self.hp.rollout_steps, continuous_actions=self.continuous_action_space)
        # self.env = PerturbationWrapper(self.env, hp.noise)

        RANDOM_SEED = 0
        torch.random.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        # torch.set_num_threads(8)


    # 通过checkpoint恢复或初始化网络参数
    def start_or_resume_from_checkpoint(self):
        max_checkpoint_iteration = get_last_checkpoint_iteration(self.base_checkpoint_path)
        # max_checkpoint_iteration = 1500
        
        if max_checkpoint_iteration <= 0:
            print("找不到checkpoint: ", self.base_checkpoint_path)
            NotImplementedError
        else:   # 从checkpoint恢复
            self.team1_actor, self.team1_critic, _, _, self.team2_actor, self.team2_critic, \
                _, _, self.hp, env_name = load_from_checkpoint(self.base_checkpoint_path, max_checkpoint_iteration, 'cpu')
            assert env_name == self.env_name, "To resume training environment must match current settings."
    

    def test(self):
        test_iteration = 10
        total_reward = np.zeros((test_iteration, self.hp.num_team1 + self.hp.num_team2))
        for k in range(test_iteration):
            episode_reward = np.zeros(self.hp.num_team1 + self.hp.num_team2)
            self.team1_actor = self.team1_actor.to(self.gather_device)
            self.team2_actor = self.team2_actor.to(self.gather_device)
            self.team1_critic = self.team1_critic.to(self.gather_device)
            self.team2_critic = self.team2_critic.to(self.gather_device)

            obsv = self.env.reset()
            terminal = torch.ones(1)
            agents = [agent for agent in self.env.agents]

            self.team1_actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.team2_actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.team1_critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.team2_critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)

            for t in range(self.hp.rollout_steps):

                actions = {}
                for i in range(self.hp.num_team1):
                    state = torch.tensor(obsv[agents[i]], dtype=torch.float32)
                    action_dist = self.team1_actor(state.unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                    action = action_dist.sample()
                    actions[agents[i]] = action.item()

                for i in range(self.hp.num_team1, self.hp.num_team1 + self.hp.num_team2):
                    state = torch.tensor(obsv[agents[i]], dtype=torch.float32)
                    action_dist = self.team2_actor(state.unsqueeze(0).unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                    action = action_dist.sample()
                    actions[agents[i]] = action.item()
                
                obsv, reward, done, _ = self.env.step(actions)
                numpy_reward = np.zeros(self.hp.num_team1 + self.hp.num_team2)
                for i in range(self.hp.num_team1 + self.hp.num_team2):
                    numpy_reward[i] = reward[agents[i]]
                episode_reward += numpy_reward
                terminal = torch.tensor(done[agents[i]]).float()
                if terminal:
                    break
                self.env.render()
                # time.sleep(0.15)

            print(f"Iteration: {k}, reward: {episode_reward}")
            total_reward[k] = episode_reward
        
        mean_reward =  total_reward.sum(axis=0) / test_iteration
        return mean_reward
