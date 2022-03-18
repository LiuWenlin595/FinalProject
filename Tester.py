from HyperParameter import HyperParameters
import torch
import numpy as np
import time

from LoadAndSave import *

from EnvWrappers import MaskVelocityWrapper, PerturbationWrapper


class Tester:
    def __init__(self,
                 env_name: str,
                 mask_velocity: bool,
                 experiment_name: str,
                 force_cpu_gather: bool = True,
                 workspace_path: str = './workspace') -> None:
        
        self.env_name = env_name
        self.mask_velocity = mask_velocity
        self.base_checkpoint_path = f'{workspace_path}/checkpoints/{experiment_name}/'
        self.start_or_resume_from_checkpoint()
        self.obsv_dim, self.action_dim, self.continuous_action_space = get_env_space(env_name)
        
        self.train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gather_device = "cuda:0" if torch.cuda.is_available() and not force_cpu_gather else "cpu"

        self.env = gym.make(self.env_name)
        if self.mask_velocity:
            self.env = MaskVelocityWrapper(self.env)
        self.env = PerturbationWrapper(self.env, hp.noise)

        RANDOM_SEED = 0
        torch.random.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        # torch.set_num_threads(8)


    # 通过checkpoint恢复或初始化网络参数
    def start_or_resume_from_checkpoint(self):
        max_checkpoint_iteration = get_last_checkpoint_iteration(self.base_checkpoint_path)
        
        if max_checkpoint_iteration <= 0:
            print("找不到checkpoint: ", self.base_checkpoint_path)
            NotImplementedError
        else:   # 从checkpoint恢复
            self.actor, self.critic, self.actor_optimizer, self.critic_optimizer, self.hp, env_name, env_mask_velocity = load_from_checkpoint(self.base_checkpoint_path, max_checkpoint_iteration, 'cpu')
            assert env_name == self.env_name, "To resume training environment must match current settings."
            assert env_mask_velocity == self.mask_velocity, "To resume training model architecture must match current settings."
    

    def test(self):
        total_reward = 0

        for i in range(10): 
            episode_reward = 0
            self.actor = self.actor.to(self.gather_device)
            self.critic = self.critic.to(self.gather_device)

            obsv = self.env.reset()
            terminal = torch.ones(1)

            self.actor.get_init_state(self.hp.parallel_rollouts, self.gather_device)
            self.critic.get_init_state(self.hp.parallel_rollouts, self.gather_device)

            for _ in range(self.hp.rollout_steps):

                state = torch.tensor([obsv], dtype=torch.float32)
                action_dist = self.actor(state.unsqueeze(0).to(self.gather_device), terminal.to(self.gather_device))
                action = action_dist.sample()
                # if not self.actor.continuous_action_space:
                #     action = action.squeeze(1)
                
                action_np = action[0].cpu().numpy()
                obsv, reward, done, _ = self.env.step(action_np)
                episode_reward += reward
                terminal = torch.tensor(done).float()
                if done:
                    break
                self.env.render()
                # time.sleep(0.01)

            print(f"Iteration: {i}, reward: {episode_reward}")
            total_reward += episode_reward
        
        mean_reward =  total_reward / 10
        return mean_reward
