import datetime
import math

from trainer import Trainer
from main import GameEnvSingleRunner
from dqn import DQNVanilla, DuellingDDQN_PRBAgent, DDQNAgent
import line_profiler

from torch.utils.tensorboard import SummaryWriter


def dqn_debug_callback_eps(trainer, *args, **kwargs):
    agent_info = kwargs["agent_info"]
    if trainer.episode % 10 == 0:
        trainer.summary_writer.add_scalar("debug/eps", agent_info["eps"], trainer.episode)


def dqn_debug_callback_weights(trainer, *args, **kwargs):
    agent = trainer.agent
    if trainer.episode % 10 == 0:
        for x in agent.dqn_net_local.named_parameters():
            if "head_nets" in x[0]:
                trainer.summary_writer.add_histogram("networks/local_" + x[0], x[1], trainer.episode)
        for x in agent.dqn_net_target.named_parameters():
            if "head_nets" in x[0]:
                trainer.summary_writer.add_histogram("networks/target_" + x[0], x[1], trainer.episode)


if __name__ == '__main__':
    DO_PROFILING = False
    from agent import agent, env, hidden_size, optim_kwargs

    # agent = DQNVanilla(env, 0.99, hidden_size, minibatch=512, optim_kwargs=optim_kwargs)
    # create string from hyper params as f string
    hyper_params_str = f"{agent.__class__.__name__}_lr_{optim_kwargs['lr']}_batch_{agent.minibatch}_layers_{hidden_size}"

    summary_writer = SummaryWriter(log_dir=f"tb_logs/{datetime.datetime.now():%D-%m-%Y}/{hyper_params_str}")
    agent.summary_writer = summary_writer
    # agent.load_weights("weights/weights_900.pt")
    trainer = Trainer(env, agent, summary_writer=summary_writer)
    # run line profiler over train code
    if DO_PROFILING:
        lp = line_profiler.LineProfiler()
        # add functions to profile
        lp.add_function(trainer._episode)
        lp_wrapper = lp(trainer.train)
        # profile simuation
        lp.add_function(trainer._episode)
        lp_wrapper(max_steps=100, num_episodes=3)
        lp.print_stats()
    else:
        trainer.train(max_steps=2000, num_episodes=5000)
