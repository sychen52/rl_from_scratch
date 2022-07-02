import gym
import torch
from torch import nn
import numpy as np
import math
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, writer: SummaryWriter, epoch_len: int):
        self.writer = writer
        self.epoch = 0
        self.epoch_len = epoch_len

    def add_scale(self, name: str, num, i_step: int = 0):
        if isinstance(num, torch.Tensor):
            num = num.item()
        self.writer.add_scalar(name, num, self.epoch * self.epoch_len + i_step)

    def __del__(self):
        self.writer.close()


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs),
                             -1)  # Critical to ensure v has right shape.


class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std
        self._dist = torch.distributions.Normal(self.mu,
                                                torch.exp(self.log_std))

    def sample(self):
        """
        Returns:
            A PyTorch Tensor of samples from the diagonal Gaussian distribution with
            mean and log_std given by self.mu and self.log_std.
        """
        return self._dist.sample()

    def log_prob(self, value):

        # when using torch.distributions.Normal, all dimension are independent
        # so I need to do the summation myself.
        return self._dist.log_prob(value).sum(axis=-1)

    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.sum(axis=-1)


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        """
        Initialize an MLP Gaussian Actor by making a PyTorch module for computing the
        mean of the distribution given a batch of observations, and a log_std parameter.

        Make log_std a PyTorch Parameter with the same shape as the action vector, 
        independent of observations, initialized to [-0.5, -0.5, ..., -0.5].
        (Make sure it's trainable!)
        """
        self.log_std = torch.nn.Parameter(torch.zeros((act_dim)) - 0.5)
        self.mu_net = mlp((obs_dim, ) + hidden_sizes + (act_dim, ), activation)
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        return pi

    def log_prob(self, obs, act):
        pi = self(obs)
        logp = pi.log_prob(act)
        return logp

    def sample_action(self, ob: np.ndarray):
        self.eval()
        with torch.no_grad():
            pi = self(
                torch.as_tensor(ob, dtype=torch.float32,
                                device=self.device).view(1, -1))
            return pi.sample().squeeze().cpu().numpy()


def sample_trajectory(env: gym.Env,
                      actor: MLPGaussianActor,
                      min_total_steps: int = 1):
    rewards: list[float] = []
    obs = []
    actions = []
    dones = []
    while len(rewards) < min_total_steps:
        done = False
        ob = env.reset()
        while not done:
            action = actor.sample_action(ob)
            obs.append(ob)
            actions.append(action)
            ob, reward, done, info = env.step(action)
            rewards.append(reward)
            dones.append(done)
    obs = np.array(obs)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    return obs, actions, rewards, dones


def compute_reward_to_go(rewards, dones, gamma=0.99):
    reward_to_go = np.zeros((rewards.shape[0] + 1), dtype=np.float32)
    for i in reversed(range(rewards.shape[0])):
        if dones[i]:
            reward_to_go[i] = rewards[i]
        else:
            reward_to_go[i] = reward_to_go[i + 1] * gamma + rewards[i]
    return reward_to_go[:-1]


def average_total_reward(rewards, dones):
    i_previous = 0
    count = 0
    cum = 0
    for i in range(dones.shape[0]):
        if dones[i]:
            cum += rewards[i_previous:i + 1].sum()
            i_previous = i + 1
            count += 1
    return cum / count


def policy_loss_ppo(obs: torch.Tensor,
                    actions: torch.Tensor,
                    logp_old: torch.Tensor,
                    advantage: torch.Tensor,
                    actor: MLPGaussianActor,
                    epsilon: float = 0.2):
    logp = actor.log_prob(obs, actions)
    ratio = torch.exp(logp - logp_old)
    first = ratio * advantage
    second = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return -torch.minimum(first, second).mean()


def policy_loss(obs: torch.Tensor,
                actions: torch.Tensor,
                logp_old: torch.Tensor,
                advantage: torch.Tensor,
                actor: MLPGaussianActor,
                epsilon: float = 0.2):
    logp = actor.log_prob(obs, actions)
    return -(logp * advantage).mean()


def ppo_one_epoch(env: gym.Env, actor: MLPGaussianActor, critic: MLPCritic,
                  opt_actor, opt_critic, n_steps: int, n_train_actor_per_step,
                  n_train_critic_per_step, logger: Logger):
    start = time.time()
    obs, actions, rewards, dones = sample_trajectory(env, actor, n_steps)
    logger.add_scale("time_env", time.time() - start)
    start = time.time()
    logger.add_scale("reward", average_total_reward(rewards, dones))
    reward_to_go = compute_reward_to_go(rewards, dones)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=actor.device)
    actions_tensor = torch.as_tensor(actions,
                                     dtype=torch.float32,
                                     device=actor.device)
    reward_to_go_tensor = torch.as_tensor(reward_to_go,
                                          dtype=torch.float32,
                                          device=critic.device)
    with torch.no_grad():
        logp_old = actor.log_prob(obs_tensor, actions_tensor)
        value_tensor = critic(obs_tensor)
    for i in range(n_train_actor_per_step):
        opt_actor.zero_grad()
        actor_loss = policy_loss(
            obs_tensor, actions_tensor, logp_old,
            reward_to_go_tensor - (value_tensor * reward_to_go_tensor.std() +
                                   reward_to_go_tensor.mean()), actor)
        logger.add_scale("actor_loss", actor_loss, i)
        actor_loss.backward()
        opt_actor.step()

    reward_to_go_normalized_tensor = (
        reward_to_go_tensor -
        reward_to_go_tensor.mean()) / reward_to_go_tensor.std()
    for i in range(n_train_critic_per_step):
        opt_critic.zero_grad()
        value_tensor = critic(obs_tensor)
        critic_loss = torch.nn.functional.mse_loss(
            value_tensor, reward_to_go_normalized_tensor)
        logger.add_scale("critic_loss", critic_loss, i)
        critic_loss.backward()
        opt_critic.step()
    logger.add_scale("time_update", time.time() - start)
    logger.epoch += 1


def main():
    n_epoches = 100
    n_steps_per_epoch = 50e3
    n_train_actor_per_step = 1
    n_train_critic_per_step = 1
    env: gym.Env = gym.make('HalfCheetah-v4')
    print(env.action_space)
    print(env.observation_space)
    obs_dim = env.observation_space.shape[0]
    actor = MLPGaussianActor(obs_dim,
                             env.action_space.shape[0],
                             hidden_sizes=(64, 64),
                             activation=torch.nn.Tanh)
    critic = MLPCritic(obs_dim, hidden_sizes=(64, 64), activation=nn.Tanh)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-2)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-2)
    logger = Logger(SummaryWriter(),
                    max(n_train_actor_per_step, n_train_critic_per_step))
    for i in range(n_epoches):
        ppo_one_epoch(env, actor, critic, opt_actor, opt_critic,
                      n_steps_per_epoch, n_train_actor_per_step,
                      n_train_critic_per_step, logger)


if __name__ == '__main__':
    main()
