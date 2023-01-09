import numpy as np
import tqdm
import torch
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, env, agent, device=torch.device("cpu"), dtype=torch.float, summary_writer=None, **kwargs):

        self.agent = agent
        self.env = env
        self.device = device
        self.dtype = dtype
        self.summary_writer = summary_writer
        self.global_steps = 0
        self.episode = 0

    # @timeit
    def train(self, num_episodes=1000, max_steps=1000, callbacks=None, silent=False, **kwargs):
        # Initialize training
        self.agent.train()
        self.agent = self.agent.to(self.device, self.dtype)
        self.num_episodes = num_episodes
        rewards, lengths, losses = [], [], []

        # Training loop
        pbar = tqdm.tqdm(range(num_episodes), disable=silent)

        for self.episode in pbar:
            # Run episode
            reward, length, loss, action_counter = self._episode(max_steps, callbacks, **kwargs)

            # Bookkeeping
            rewards.append(reward)
            losses.append(loss)
            lengths.append(length)
            self._report_episode(pbar, rewards, losses, lengths, action_counter)
            # save model weights every 100 episodes
            if self.episode % 100 == 0:
                self.agent.save_weights("weights/weights_{}.pt".format(self.episode))
                self.agent.save_weights("weights/weights_latest.pt")

        if self.summary_writer:
            self.summary_writer.close()
        self.agent.save_weights("weights/weights_final.pt")

        return np.array(rewards), np.array(lengths), np.array(losses)

    def _episode(self, max_steps, callbacks=None, **kwargs):
        # Initialize episode
        episode_reward = 0.0
        episode_length = 0
        loss = 0.0

        state = self.env.reset()
        state = torch.from_numpy(state).to(dtype=self.dtype, device=self.device)
        reward = 0.0

        # Run episode
        # create action counter to keep track of how many times each action is taken
        action_counter = []
        for episode_length in range(max_steps):
            # Agent step
            action, agent_info = self.agent(state)
            action_counter.append(action)
            # Environment step
            next_state, next_reward, done, info = self.env.step(action)
            next_state = torch.from_numpy(next_state).to(dtype=self.dtype, device=self.device)

            # Replay buffer and model updates
            curr_loss = self.agent.learn(
                state=state,
                reward=reward,
                action=action,
                done=done,
                next_state=next_state,
                next_reward=next_reward,
                episode_end=(done or episode_length == max_steps - 1),
                num_episode=self.episode,
                **agent_info,
                **kwargs,
            )
            loss += curr_loss
            # Prepare for next step
            state = next_state
            reward = next_reward

            # Book keeping and callbacks
            episode_reward += next_reward
            self.global_steps += 1
            """
            callbacks and _report step are kind of identical, I would opt for removing callbacks, and instead overwriting inheritated
            _report_steps, or is callbacks more general? It has the advantage of being easily removed, by not defining callbacks, 
            but the same can be achieved by setting report self.report_steps to False (default).
            """
            if callbacks is not None:
                for callback in callbacks:
                    callback(
                        self,
                        episode_length,
                        state=state,
                        reward=reward,
                        action=action,
                        next_state=next_state,
                        next_reward=next_reward,
                        done=done,
                        info=info,
                        agent_info=agent_info,
                    )

            if done:
                break

        return episode_reward, episode_length + 1, loss, action_counter

    def _report_episode(self, pbar, rewards, losses, lengths, action_counter, avg_length=10):
        # Tensorboard logging
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("Performance/reward", rewards[-1], self.episode)
            try:
                self.summary_writer.add_scalar("Performance/loss", losses[-1], self.episode)
            except AssertionError:
                for i, loss in enumerate(losses[-1]):
                    self.summary_writer.add_scalar(f"Performance/loss_{i}", loss, self.episode)
                self.summary_writer.add_scalar(f"Performance/loss_sum", np.sum(losses[-1]), self.episode)
            self.summary_writer.add_scalar("Performance/max_reward", np.max(np.array(rewards)), self.episode)
            self.summary_writer.add_scalar("Performance/episode_length", lengths[-1], self.episode)
            self.summary_writer.add_histogram("Debug/Actions", np.array(action_counter), self.episode)

        # Progress bar
        pbar.set_description(
            f"Episode {self.episode + 1}: reward {rewards[-1]} (last {avg_length} avg. {np.mean(rewards[-avg_length:]):.2e}")
