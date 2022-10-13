import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class DecoupledSquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.activation = activation
        self.act_dim = act_dim

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])

        self.mu_layer_list = nn.ModuleList([nn.Linear(hidden_sizes[-1], 1) for _ in range(act_dim)])
        self.log_std_layer = nn.ModuleList([nn.Linear(hidden_sizes[-1], 1) for _ in range(act_dim)])
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = self.activation()(self.linear_list2[i](x))

            mu_temp = self.mu_layer_list[i](x)
            if len(mu_temp.size()) == 1:
                mu_temp = torch.unsqueeze(mu_temp, dim=1)
            log_std_temp = self.log_std_layer[i](x)
            if len(log_std_temp.size()) == 1:
                log_std_temp = torch.unsqueeze(log_std_temp, dim=1)

            if i == 0:
                mu = mu_temp
                log_std = log_std_temp
            else:
                mu = torch.cat([mu, mu_temp], dim=1)
                log_std = torch.cat([log_std, log_std_temp], dim=1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class DecoupledBiRnnSquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled):
        super().__init__()
        self.activation = activation
        self.act_dim = act_dim
        self.rnn_type = decoupled

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])

        self.mu_layer_list = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, 1) for _ in range(act_dim)])
        self.log_std_layer = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, 1) for _ in range(act_dim)])

        if decoupled == 2:
            self.rnn_linear = nn.GRU(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        else:
            self.rnn_linear = nn.LSTM(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        for name, param in self.rnn_linear.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)


        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = torch.unsqueeze(self.activation()(self.linear_list2[i](x)), dim=0)
            if i == 0:
                x_all = x
            else:
                x_all = torch.cat([x_all, x], dim=0)
        if len(x_all.size()) == 2:
            x_all = torch.unsqueeze(x_all, dim=1)
        if self.rnn_type == 2 or self.rnn_type == 4:
            out, _ = self.rnn_linear(x_all)
        else:
            out, (_, _) = self.rnn_linear(x_all)

        for j in range(self.act_dim):
            rnn_out_dim_j = out[j, :, :]
            mu_temp = self.mu_layer_list[j](rnn_out_dim_j)
            if len(mu_temp.size()) == 1:
                mu_temp = torch.unsqueeze(mu_temp, dim=1)
            log_std_temp = self.log_std_layer[j](rnn_out_dim_j)
            if len(log_std_temp.size()) == 1:
                log_std_temp = torch.unsqueeze(log_std_temp, dim=1)

            if j == 0:
                mu = mu_temp
                log_std = log_std_temp
            else:
                mu = torch.cat([mu, mu_temp], dim=1)
                log_std = torch.cat([log_std, log_std_temp], dim=1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, decoupled, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        if decoupled == 0:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled == 1:
            self.pi = DecoupledSquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled >= 2:
            self.pi = DecoupledBiRnnSquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
