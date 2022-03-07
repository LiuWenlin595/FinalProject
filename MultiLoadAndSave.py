import os
import pickle
import torch
from typing import Optional
from dotmap import DotMap
import pathlib
import gym
from pettingzoo.mpe import simple_tag_v2

from Actor import Actor
from Critic import Critic


# 得到checkpoint最后的iteration次数
def get_last_checkpoint_iteration(base_checkpoint_path: str) -> int:
    if os.path.isdir(base_checkpoint_path):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(base_checkpoint_path)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration


# 返回checkpoint中的模型和参数
def load_checkpoint(base_checkpoint_path: str, iteration: int, map_loacation: Optional[str] = None):
    base_checkpoint = base_checkpoint_path + f"{iteration}/"
    with open(base_checkpoint + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)

    team1_actor_state_dict = torch.load(base_checkpoint + "team1_actor.pt", map_location=torch.device(map_loacation))
    team1_critic_state_dict = torch.load(base_checkpoint + "team1_critic.pt", map_location=torch.device(map_loacation))
    team1_actor_optimizer_state_dict = torch.load(base_checkpoint + "team1_actor_optimizer.pt", map_location=torch.device(map_loacation))
    team1_critic_optimizer_state_dict = torch.load(base_checkpoint + "team1_critic_optimizer.pt", map_location=torch.device(map_loacation))
    team2_actor_state_dict = torch.load(base_checkpoint + "team2_actor.pt", map_location=torch.device(map_loacation))
    team2_critic_state_dict = torch.load(base_checkpoint + "team2_critic.pt", map_location=torch.device(map_loacation))
    team2_actor_optimizer_state_dict = torch.load(base_checkpoint + "team2_actor_optimizer.pt", map_location=torch.device(map_loacation))
    team2_critic_optimizer_state_dict = torch.load(base_checkpoint + "team2_critic_optimizer.pt", map_location=torch.device(map_loacation))
    
    return (team1_actor_state_dict, team1_critic_state_dict, team1_actor_optimizer_state_dict, team1_critic_optimizer_state_dict,
           team2_actor_state_dict, team2_critic_state_dict, team2_actor_optimizer_state_dict, team2_critic_optimizer_state_dict, checkpoint.hp, checkpoint.env)


# 得到gym_env的基本状态空间和动作空间
def get_env_space(env_name: str, hp):
    env = simple_tag_v2.parallel_env(num_good=hp.num_team2, num_adversaries=hp.num_team1, num_obstacles=hp.num_obstacles, max_cycles=hp.rollout_steps, continuous_actions=False)
    action_space = env.action_space(env.possible_agents[0])
    continuous_action_space = type(action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim =  action_space.shape[0]
    else:
        action_dim = action_space.n 
    global_obsv_dim= env.state_space.shape[0]
    team1_obsv_dim = env.observation_space(env.possible_agents[0]).shape[0]
    team2_obsv_dim = env.observation_space(env.possible_agents[-1]).shape[0]
    return global_obsv_dim, team1_obsv_dim, team2_obsv_dim, action_dim, continuous_action_space


# 通过checkpoint中的模型和参数, 拷贝一个新的模型用于训练
def load_from_checkpoint(base_checkpoint_path: str, iteration: int, map_loacation: Optional[str] = None):
    
    team1_actor_state_dict, team1_critic_state_dict, team1_actor_optimizer_state_dict, team1_critic_optimizer_state_dict, team2_actor_state_dict, \
        team2_critic_state_dict, team2_actor_optimizer_state_dict, team2_critic_optimizer_state_dict, hp, env_name = load_checkpoint(base_checkpoint_path, iteration, map_loacation)
    
    global_obsv_dim, team1_obsv_dim, team2_obsv_dim, action_dim, continuous_action_space = get_env_space(env_name)

    team1_actor = Actor(team1_obsv_dim,
                action_dim,
                continuous_action_space=continuous_action_space,
                hp = hp)
    team2_actor = Actor(team2_obsv_dim,
                action_dim,
                continuous_action_space=continuous_action_space,
                hp = hp)
    team1_critic = Critic(global_obsv_dim, hp)
    team2_critic = Critic(global_obsv_dim, hp)

    team1_actor_optimizer = torch.optim.AdamW(team1_actor.parameters(), lr=hp.actor_learning_rate)
    team1_critic_optimizer = torch.optim.AdamW(team1_critic.parameters(), lr=hp.critic_learning_rate)
    team2_actor_optimizer = torch.optim.AdamW(team2_actor.parameters(), lr=hp.actor_learning_rate)
    team2_critic_optimizer = torch.optim.AdamW(team2_critic.parameters(), lr=hp.critic_learning_rate)
    
    # 从预训练模型中加载权重
    team1_actor.load_state_dict(team1_actor_state_dict, strict=True) 
    team1_critic.load_state_dict(team1_critic_state_dict, strict=True)
    team1_actor_optimizer.load_state_dict(team1_actor_optimizer_state_dict)
    team1_critic_optimizer.load_state_dict(team1_critic_optimizer_state_dict)
    team2_actor.load_state_dict(team2_actor_state_dict, strict=True) 
    team2_critic.load_state_dict(team2_critic_state_dict, strict=True)
    team2_actor_optimizer.load_state_dict(team2_actor_optimizer_state_dict)
    team2_critic_optimizer.load_state_dict(team2_critic_optimizer_state_dict)

    for state in team1_actor_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    for state in team1_critic_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    for state in team2_actor_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    for state in team2_critic_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    return team1_actor, team1_critic, team1_actor_optimizer, team1_critic_optimizer, \
        team2_actor, team2_critic, team2_actor_optimizer, team2_critic_optimizer, hp, env_name


# 保存checkpoint
def save_checkpoint(base_checkpoint_path: str, team1_actor, team1_critic, team1_actor_optimizer, team1_critic_optimizer, 
                team2_actor, team2_critic, team2_actor_optimizer, team2_critic_optimizer, iteration, hp, env_name):
    checkpoint = DotMap()
    checkpoint.env = env_name
    checkpoint.iteration = iteration
    checkpoint.hp = hp
    base_checkpoint = base_checkpoint_path + f"{iteration}/"
    pathlib.Path(base_checkpoint).mkdir(parents=True, exist_ok=True) 
    with open(base_checkpoint + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    
    torch.save(team1_actor.state_dict(), base_checkpoint + "team1_actor.pt")
    torch.save(team1_critic.state_dict(), base_checkpoint + "team1_critic.pt")
    torch.save(team1_actor_optimizer.state_dict(), base_checkpoint + "team1_actor_optimizer.pt")
    torch.save(team1_critic_optimizer.state_dict(), base_checkpoint + "team1_critic_optimizer.pt")
    torch.save(team2_actor.state_dict(), base_checkpoint + "team2_actor.pt")
    torch.save(team2_critic.state_dict(), base_checkpoint + "team2_critic.pt")
    torch.save(team2_actor_optimizer.state_dict(), base_checkpoint + "team2_actor_optimizer.pt")
    torch.save(team2_critic_optimizer.state_dict(), base_checkpoint + "team2_critic_optimizer.pt")
