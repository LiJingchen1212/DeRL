import numpy as np
import scipy.signal

import torch
import torch.nn as nn


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


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class DecoupledBiRnnMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled):
        super().__init__()

        self.activation = activation
        self.act_dim = act_dim
        self.rnn_type = decoupled

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])

        if decoupled == 2:
            self.rnn_linear = nn.GRU(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        else:
            self.rnn_linear = nn.LSTM(input_size=list(hidden_sizes)[1], hidden_size=list(hidden_sizes)[1], num_layers=2, bidirectional=True)
        for name, param in self.rnn_linear.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

        self.linear_list3 = nn.ModuleList([nn.Linear(list(hidden_sizes)[1] * 2, 1) for _ in range(act_dim)])

        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
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
            linear3_out_temp = nn.Tanh()(self.linear_list3[j](rnn_out_dim_j))
            if len(linear3_out_temp.size()) == 1:
                linear3_out_temp = torch.unsqueeze(linear3_out_temp, dim=1)

            if j == 0:
                linear3_out = linear3_out_temp
            else:
                linear3_out = torch.cat([linear3_out, linear3_out_temp], dim=1)

        return self.act_limit * linear3_out


class DecoupledMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()

        self.activation = activation
        self.act_dim = act_dim

        self.linear_list1 = nn.ModuleList([nn.Linear(obs_dim, list(hidden_sizes)[0]) for _ in range(act_dim)])
        self.linear_list2 = nn.ModuleList([nn.Linear(list(hidden_sizes)[0], list(hidden_sizes)[1]) for _ in range(act_dim)])
        self.linear_list3 = nn.ModuleList([nn.Linear(list(hidden_sizes)[1], 1) for _ in range(act_dim)])

        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        for i in range(self.act_dim):
            x = self.activation()(self.linear_list1[i](obs))
            x = self.activation()(self.linear_list2[i](x))
            out_temp = nn.Tanh()(self.linear_list3[i](x))

            if len(out_temp.size()) == 1:
                out_temp = torch.unsqueeze(out_temp, dim=1)

            if i == 0:
                out = out_temp
            else:
                out = torch.cat([out, out_temp], dim=1)
        return self.act_limit * out


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, decoupled, hidden_sizes=(64,64),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        if decoupled == 0:
            self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled == 1:
            self.pi = DecoupledMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif decoupled >= 2:
            self.pi = DecoupledBiRnnMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, decoupled)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()
