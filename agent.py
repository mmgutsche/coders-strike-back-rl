import math

from dqn import DuellingDDQN_PRBAgent, D3QN_PRB_NStep, DQNVanilla, DDQNAgent, DDQN_PRBAgent
from main import GameEnvSingleRunner

env = GameEnvSingleRunner()
hidden_size = (128,)
optim_kwargs = {"lr": 1e-3}


def eps_func_decay(episode):
    rel_val = episode
    decay_rate = 0.001
    eps = 1 * math.exp(-rel_val * decay_rate)
    eps = max(eps, 0.05)
    return eps


def eps_func_linear(episode):
    eps = 1 - (episode / 1000)
    eps = max(eps, 0.05)
    return eps


agent = DDQN_PRBAgent(env, 0.9, hidden_size,
                      minibatch=2024,
                      optim_kwargs=optim_kwargs,
                      history_length=1024,
                      target_update_interval=50,
                      eps_func=eps_func_decay)
