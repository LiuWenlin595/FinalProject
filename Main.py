import argparse
from ast import arg
import datetime
import imp

from HyperParameter import HyperParameters
from Trainer import Trainer
from MultiTrainer import MultiTrainer
from Tester import Tester


def make_hp(args) -> HyperParameters:
    if args.env == "CartPole-v1" and args.mask_velocity:
        # Working perfectly with patience.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=512, batch_size=128, recurrent_seq_len=8, patience=200)
    elif args.env == "CartPole-v1" and not args.mask_velocity:
        # Working perfectly with patience.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=512, batch_size=128, recurrent_seq_len=8, patience=200)
    elif args.env == "Pendulum-v0" and args.mask_velocity:
        # Works well.     
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=200, batch_size=512, recurrent_seq_len=8,
                            init_log_std_dev=1., trainable_std_dev=True, actor_learning_rate=1e-3, critic_learning_rate=1e-3)
    elif args.env == "LunarLander-v2" and args.mask_velocity:
        # Works well.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=512, recurrent_seq_len=8, patience=1000) 
    elif args.env == "LunarLanderContinuous-v2" and args.mask_velocity:
        # Works well. 
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=200)
    elif args.env == "LunarLanderContinuous-v2" and not args.mask_velocity:
        # Works well.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=100)
    elif args.env == "BipedalWalker-v3" and not args.mask_velocity:
        # Working :-D
        hp = HyperParameters(parallel_rollouts=8, rollout_steps=2048, batch_size=256, patience=1000, entropy_factor=1e-4,
                            init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1.)
                            #init_log_std_dev=1., trainable_std_dev=True)
    elif args.env == "BipedalWalkerHardcore-v3" and not args.mask_velocity:
        # Working :-D
        hp = HyperParameters(batch_size=1024, parallel_rollouts=16, recurrent_seq_len=8, rollout_steps=1024, patience=5000, entropy_factor=1e-4, 
                            init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1., hidden_size=256)
    elif args.env == "simple_tag":
        # Working :-D
        hp = HyperParameters(batch_size=128, parallel_rollouts=1, recurrent_seq_len=4, num_episodes=100, rollout_steps=25, patience=5000,
                            entropy_factor=1e-4, hidden_size=128, num_team1=3, num_team2=2, num_obstacles=2)
    else:
        raise NotImplementedError  
    
    hp.use_lstm = args.use_lstm
    hp.noise = args.noise
    hp.sample = args.sample
    return hp


def train(args):
    start_time = datetime.datetime.now()
    hp = make_hp(args)
    experiment_name = f'{args.env}_{"LSTM" if args.use_lstm else "NoLSTM"}_{"NoVelocity" if args.mask_velocity else "Velocity"}_noise{args.noise}_sample{args.sample}'
    trainer = Trainer(args.env, args.mask_velocity, experiment_name, hp)
    score = trainer.train()
    end_time = datetime.datetime.now()
    print("max reward: ", score)
    print("spend time: ", (end_time-start_time).seconds)
    print("end time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def multi_train(args):
    start_time = datetime.datetime.now()
    hp = make_hp(args)
    experiment_name = f'{args.env}_{"LSTM" if args.use_lstm else "NoLSTM"}_{"NoVelocity" if args.mask_velocity else "Velocity"}_noise{args.noise}_sample{args.sample}'
    trainer = MultiTrainer(args.env, experiment_name, hp)
    team1_score, team2_score = trainer.train()
    end_time = datetime.datetime.now()
    print("max reward: ", team1_score, team2_score)
    print("spend time: ", (end_time-start_time).seconds)
    print("end time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def test(args):
    start_time = datetime.datetime.now()
    hp = make_hp(args)
    experiment_name = f'{args.env}_{"LSTM" if args.use_lstm else "NoLSTM"}_{"NoVelocity" if args.mask_velocity else "Velocity"}_noise{args.noise}_sample{args.sample}'
    tester = Tester(args.env, args.mask_velocity, experiment_name, hp)
    score = tester.test()
    end_time = datetime.datetime.now()
    print("test reward: ", score)
    print("spend time: ", (end_time-start_time).seconds / 3600)
    print("end time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default='simple_tag')
    # parser.add_argument("-e", "--env", type=str, default='BipedalWalkerHardcore-v3')
    parser.add_argument("-m", "--mask-velocity", default=False)
    parser.add_argument("-n", "--name", type=str, default='experiment')
    parser.add_argument("-R", "--use-lstm", default=True)
    parser.add_argument("-s", "--sample", default=0)
    parser.add_argument("--noise", type=float, default=0.0)

    args = parser.parse_args()

    multi_train(args)
    # train(args)
    # test(args)