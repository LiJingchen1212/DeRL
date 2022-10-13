import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


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


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class DecoupledMLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.activation = activation
        self.act_dim = act_dim

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])
        self.linear_list3 = nn.ModuleList([nn.Linear(list(hidden_sizes)[1], 1) for _ in range(act_dim)])

    def _distribution(self, obs):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = self.activation()(self.linear_list2[i](x))
            out_temp = nn.Identity()(self.linear_list3[i](x))

            if len(out_temp.size()) == 1:
                out_temp = torch.unsqueeze(out_temp, dim=1)

            if i == 0:
                mu = out_temp
            else:
                mu = torch.cat([mu, out_temp], dim=1)

        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class DecoupledBiRnnMLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, decoupled):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.activation = activation
        self.act_dim = act_dim
        self.rnn_type = decoupled

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])
        self.linear_list3 = nn.ModuleList([nn.Linear(list(hidden_sizes)[1] * 2, 1) for _ in range(act_dim)])

        if decoupled == 2:
            self.rnn_linear = nn.GRU(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        else:
            self.rnn_linear = nn.LSTM(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)

        for name, param in self.rnn_linear.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def _distribution(self, obs):
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            linear_out_temp = torch.unsqueeze(self.activation()(self.linear_list2[i](x)), dim=0)

            if i == 0:
                linear_out = linear_out_temp
            else:
                linear_out = torch.cat([linear_out, linear_out_temp], dim=0)

        if len(linear_out.size()) == 2:
            linear_out = torch.unsqueeze(linear_out, dim=1)
        if self.rnn_type == 2:
            out, _ = self.rnn_linear(linear_out)
        else:
            out, (_, _) = self.rnn_linear(linear_out)

        for j in range(self.act_dim):
            rnn_out_dim_j = out[j, :, :]
            linear3_out_temp = nn.Identity()(self.linear_list3[j](rnn_out_dim_j))
            if len(linear3_out_temp.size()) == 1:
                linear3_out_temp = torch.unsqueeze(linear3_out_temp, dim=1)

            if j == 0:
                mu = linear3_out_temp
            else:
                mu = torch.cat([mu, linear3_out_temp], dim=1)

        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, decoupled,
                 hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            if decoupled == 0:
                self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            elif decoupled == 1:
                self.pi = DecoupledMLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
            elif decoupled >= 2:
                self.pi = DecoupledBiRnnMLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, decoupled)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
