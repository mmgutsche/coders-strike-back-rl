import gym
from torch.utils.tensorboard import SummaryWriter
import math

from dqn import DuellingDDQN_PRBAgent, D3QN_PRB_NStep, DQNVanilla, DDQNAgent, DDQN_PRBAgent

from trainer import Trainer

env = gym.envs.make("CartPole-v1")
summary_writer = SummaryWriter(log_dir=f"tb_logs/cartpole_test")

hidden_size = (64,)
optim_kwargs = {"lr": 1e-3}

MAX_EPISODES = 1000

agent = DDQNAgent(env, 0.9, hidden_size,
                  minibatch=8,
                  optim_kwargs=optim_kwargs,
                  history_length=20,
                  target_update_interval=10, epsilon_decay=0.99, epsilon=0.3
                  )

# agent = DQNVanilla(env, 0.99, hidden_size, minibatch=512, optim_kwargs=optim_kwargs)


trainer = Trainer(env, agent, summary_writer=summary_writer)
trainer.train(num_episodes=1000)
