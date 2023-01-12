"""
Different kind of DQNs
vanilla DQN -> Q(s,a) = R + gamma * max[ Q(s_next, a_next')]
Double DQN ->  Q(s,a) = R + gamma * Q'(s_next, argmax Q(s_next, a_next'))
"""
from typing import Protocol

import torch
from torch import nn

import random
import numpy as np
import logging

import math

from nets import MultiHeadedMLP, DuellingDQNNet, CNN, NoisyLinear, NoisyFactorizedLinear
from replay_buffer import History, PriorityReplayBuffer

logger = logging.getLogger("dqn_agent")
# constants
STATE = "state"
ACTION = "action"
REWARD = "reward"
NEXT_STATE = "next_state"
DONE = "done"


class DQNVanilla(nn.Module):
    """ Deep Q-learning """

    def __init__(self, env, gamma, hidden_sizes=(64,), activation=nn.ReLU(), n_steps=1, epsilon=0.3, epsilon_decay=0.99,
                 minibatch=64, summary_writer=None, optim_kwargs=None,
                 history_length=None, dtype=torch.float, device=torch.device("cpu")):
        super().__init__()

        self.gamma = gamma
        self.device = device
        self.dtype = dtype
        self.action_space = env.action_space
        self.num_actions = self.action_space.n
        self.state_length = env.observation_space.shape[0]
        self._init_replay_buffer(history_length)
        self.optimizer = None
        self.optim_kwargs = optim_kwargs
        self.gamma = gamma

        self.summary_writer = summary_writer
        self.n_steps = n_steps  # for easier n_step inheritance
        self.minibatch = minibatch
        self.eps = epsilon
        self.step = 0
        self.eps_decay = epsilon_decay
        self._setup_networks(hidden_sizes, activation)

    # self._init_optimizer()

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation,
            head_activations=(None,)
        )

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            optim_kwargs = {} if self.optim_kwargs is None else self.optim_kwargs
            self.optimizer = torch.optim.Adam(params=self.parameters(), **optim_kwargs)
        self.dqn_net_local.train()

    def _init_replay_buffer(self, history_length):
        self.history = History(max_length=history_length, dtype=self.dtype, device=self.device)

    def _min_history_length(self):
        return self.minibatch

    def _store(self, **kwargs):
        self.history.store(**kwargs)

    def _finish_episode(self):
        ...

    def learn(self, state, reward, action, done, next_state, next_reward, episode_end, num_episode, *args, **kwargs):

        self.step += 1

        loss = 0

        if self.summary_writer is not None and episode_end:
            self.summary_writer.add_scalar("debug/eps", self.eps, num_episode)
        self._store(state=state, reward=reward, action=action, next_state=next_state, done=done)

        if len(self.history) > self._min_history_length():
            loss = self._learn()

        if done:
            self._finish_episode()
        self.eps = max(self.eps * self.eps_decay, 0.01)
        return np.array([loss])

    def _learn(self):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        memory_idx, importance_sampling_weights, experiences = self.history.sample(n=self.minibatch)
        experiences = self._prep_minibatch(experiences)  # non_final_next_states are only needed for n-step variants

        states = experiences[STATE]
        actions = experiences[ACTION]
        rewards = experiences[REWARD]
        next_states = experiences[NEXT_STATE]
        dones = experiences[DONE]

        q = self.dqn_net_local(states)[0]

        q_eval = q.gather(1, actions.view(-1, 1))  # shape (batch, 1)
        q_target = self._calc_reward(next_states, rewards, dones)
        q_eval = torch.squeeze(q_eval)
        q_target = torch.squeeze(q_target)

        # ------------------- update target network ------------------- #
        self._update_memory(q_eval.detach(), q_target.detach(), memory_idx)
        loss = self._optimize_loss(q_eval, q_target, importance_sampling_weights)

        return loss.item()

    def _prep_minibatch(self, experiences):
        for key, item in experiences.items():
            if isinstance(item, list) and isinstance(item[0], (float, int, bool)):
                experiences[key] = torch.Tensor(item).to(self.device)
            elif isinstance(item, list) and isinstance(item[0], np.number):
                experiences[key] = torch.from_numpy(np.array(item)).to(self.device)
            elif isinstance(item, np.ndarray):
                experiences[key] = torch.from_numpy(item).to(self.device)
            elif isinstance(item, torch.Tensor):
                experiences[key] = item.squeeze().to(self.device)
            elif isinstance(item[0], torch.Tensor):
                experiences[key] = torch.stack(item, dim=0).to(self.device)
            # elif key == "action":
            #     experiences[key] = torch.from_numpy(item).long().to(self.device)
            # elif isinstance(item[0], torch.Tensor):
            #     experiences[key] = torch.cat(item)
            else:
                raise NotImplementedError(f"Unkown type {type(item[0])}")
        return experiences

    def _optimize_loss(self, q_eval, q_target, weights, grad_clip_value=100.0):
        criterion = torch.nn.MSELoss()
        loss = criterion(q_eval, q_target).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_value)
        self.optimizer.step()
        return loss

    def _update_memory(self, target_old, target, memory_idx):
        ...

    def _calc_reward(self, next_states, rewards, done):
        """
        Reward function for vanilla DQN
        """
        with torch.no_grad():
            done_mask = done == 1
            q_predict = self.dqn_net_local(next_states)[0].detach()
            next_max_q_values, indices = torch.max(q_predict, axis=-1)
            next_max_q_values = next_max_q_values * ~done_mask

            q_targets = rewards + self.gamma ** self.n_steps * next_max_q_values

        return q_targets

    def _action_selection(self, scores):
        """epsilon greedy slection"""
        if random.random() > self.eps:
            value = np.max(scores.cpu().data.numpy())
            action = np.argmax(scores.cpu().data.numpy())
            logger.debug(f"action: {action}. eps: {self.eps}, value: {value}")
        else:
            action = random.choice(np.arange(self.num_actions))
            value = -1
            logger.debug(f"random action: {action}. eps: {self.eps}")

        return action, {"value": value, "eps": self.eps}

    def forward(self, state, **kwargs):
        """
        Given an environment state, pick the next action and return it together with the log likelihood and an estimate of the state value.


        """

        self.dqn_net_local.eval()
        with torch.no_grad():
            scores = self.dqn_net_local(state)[0]
        self.dqn_net_local.train()
        return self._action_selection(scores)
        # Epsilon -greedy action selection

    def save_weights(self, path):
        torch.save(self.dqn_net_local.state_dict(), path)

    def load_weights(self, path):
        self.dqn_net_local.load_state_dict(torch.load(path))


class Noisy_DQN(DQNVanilla):
    # from https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py
    def __init__(self, *args, noisy_init_sigma=0.5, **kwargs):
        self.noisy_init_sigma = noisy_init_sigma
        super().__init__(*args, **kwargs)

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length,
            hidden_sizes=hidden_sizes,
            head_sizes=(self.num_actions,),
            activation=activation,
            head_activations=(None,),
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )

    def _action_selection(self, scores):
        """argmax selection"""

        value = np.max(scores.cpu().data.numpy())
        action = np.argmax(scores.cpu().data.numpy())
        return action, {"value": value}


class DQN_NStep_Agent(DQNVanilla):
    # modified from https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb
    def __init__(self, *args, **kwargs):
        self.n_step_buffer = []
        super().__init__(*args, **kwargs)

    def _store(self, **kwargs):
        self.n_step_buffer.append(kwargs)
        if len(self.n_step_buffer) < self.n_steps:
            return
        R = sum([self.n_step_buffer[i]["reward"] * (self.gamma ** i) for i in range(self.n_steps)])
        n_step_experience = self.n_step_buffer.pop(0)
        n_step_experience[REWARD] = R
        n_step_experience[NEXT_STATE] = kwargs[NEXT_STATE]
        self.history.store(**n_step_experience)

    def _finish_episode(self):
        last_experience = self.n_step_buffer[-1]
        while len(self.n_step_buffer) > 0:
            R = sum([self.n_step_buffer[i][REWARD] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
            n_step_experience = self.n_step_buffer.pop(0)
            n_step_experience[REWARD] = R
            n_step_experience[NEXT_STATE] = last_experience[NEXT_STATE]
            # check, technically done references to the current state, not the next state. But in calculations it makes
            # no difference, since it is only a mask to remove "invalid states"
            # check if needed at all, if we are not using memories at the end of an episode, why bother saving them? Only for q eval of current state?!
            n_step_experience[DONE] = True
            self.history.store(**n_step_experience)


class DDQNAgent(DQNVanilla):
    def __init__(self, *args, target_update_interval=4, tau=1e-2, **kwargs):

        self.tau = tau
        self.target_update_interval = target_update_interval
        super().__init__(*args, **kwargs)

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation,
            head_activations=(None,)
        )

        self.dqn_net_target = MultiHeadedMLP(
            self.state_length, hidden_sizes=hidden_sizes, head_sizes=(self.num_actions,), activation=activation,
            head_activations=(None,)
        )
        self.dqn_net_target.eval()
        # have same params for local and target net
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)

    def _calc_reward(self, next_states, rewards, done):
        """
        Reward function for double DQN
        """
        done_mask = done == 1
        done_mask = done_mask.squeeze()
        with torch.no_grad():
            next_q_values_target = self.dqn_net_target(next_states)[0].detach()
            next_q_values_local = self.dqn_net_local(next_states)[0].detach()
            idx = torch.argmax(next_q_values_local, axis=-1).squeeze()  # get action from local net

            q_term = next_q_values_target[torch.arange(
                next_q_values_target.shape[0]), idx]  # get Q value associate with this action from target net
            q_term = q_term * ~done_mask
            q_targets = rewards + self.gamma ** self.n_steps * q_term
        if (self.step % self.target_update_interval) == 0:
            self.soft_update(self.dqn_net_local, self.dqn_net_target, self.tau)

        return q_targets

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        logger.debug("Updating target net")
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


class DQN_PRBAgent(DQNVanilla):
    def _init_replay_buffer(self, history_length):
        self.history = PriorityReplayBuffer(history_length, device=self.device, dtype=self.dtype)

    def _update_memory(self, target_old, target, memory_idx):
        abs_errs = torch.abs(target_old - target).view(memory_idx.shape)
        self.history.batch_update(memory_idx, abs_errs)

    def _optimize_loss(self, q_eval, q_target, weights, grad_clip_value=100.0):
        weights = torch.from_numpy(weights).to(self.device, self.dtype)
        loss = weights * (q_eval - q_target) ** 2
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class DuellingDQNAgent(DQNVanilla):
    """
    network is split into output for V and A, which are related
    by Q = V + A, which is either represented
    as Q = V +(A - max_a(A)) or
    as Q = V +(A - mean(A)) to avoid the "unidentifiablity", meaning the recovery of V and A unambigously from Q
    """

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions,
                                            activation=activation)


class DuellingDDQNAgent(DDQNAgent):
    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions,
                                            activation=activation)
        self.dqn_net_target = DuellingDQNNet(self.state_length, hidden_sizes=hidden_sizes, out_size=self.num_actions,
                                             activation=activation)
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)  # tau = 1 corresponds to a hard update


class Noisy_D3QN(DuellingDDQNAgent, Noisy_DQN):
    # from https://github.com/Shmuma/ptan/blob/master/samples/rainbow/lib/dqn_model.py

    def _setup_networks(self, hidden_sizes, activation):
        self.dqn_net_local = DuellingDQNNet(
            self.state_length,
            hidden_sizes=hidden_sizes,
            out_size=self.num_actions,
            activation=activation,
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )
        self.dqn_net_target = DuellingDQNNet(
            self.state_length,
            hidden_sizes=hidden_sizes,
            out_size=self.num_actions,
            activation=activation,
            linear=NoisyFactorizedLinear,
            init_sigma=self.noisy_init_sigma,
        )
        self.soft_update(self.dqn_net_local, self.dqn_net_target, 1)  # tau = 1 corresponds to a hard update


class DDQN_PRBAgent(DQN_PRBAgent, DDQNAgent):
    pass


class DuellingDDQN_PRBAgent(DQN_PRBAgent, DuellingDDQNAgent):
    pass


class D3QN_PRB_NStep(DQN_NStep_Agent, DuellingDDQN_PRBAgent):
    pass


class Noisy_D3QN_PRB_NStep(Noisy_D3QN, D3QN_PRB_NStep):
    pass
