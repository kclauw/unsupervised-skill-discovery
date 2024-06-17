import numpy as np
import torch
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax

from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from utils.replay_buffer import ReplayBuffer, Transition


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)

class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)

class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob

class SACAgent:
    def __init__(self,
                 p_z,
                 cfg):
        self.cfg = cfg
       
        self.parameters = cfg.agent.parameters
        self.p_z = np.tile(p_z, self.parameters.batch_size).reshape(self.parameters.batch_size, self.cfg.n_skills)
        self.replay_buffer = ReplayBuffer(self.parameters.mem_size, self.cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
     
        torch.manual_seed(self.cfg.seed)
        
        self.policy_network = PolicyNetwork(n_states=self.parameters.obs_shape + self.cfg.n_skills,
                                            n_actions=self.parameters.action_shape,
                                            action_bounds=self.parameters.action_bounds,
                                            n_hidden_filters=self.parameters.n_hidden).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.parameters.obs_shape + self.cfg.n_skills,
                                              n_actions=self.parameters.action_shape,
                                              n_hidden_filters=self.parameters.n_hidden).to(self.device)
        
        self.q_value_network2 = QvalueNetwork(n_states=self.parameters.obs_shape + self.cfg.n_skills,
                                              n_actions=self.parameters.action_shape,
                                              n_hidden_filters=self.parameters.n_hidden).to(self.device)

        self.value_network = ValueNetwork(n_states=self.parameters.obs_shape + self.cfg.n_skills,
                                          n_hidden_filters=self.parameters.n_hidden).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.parameters.obs_shape + self.cfg.n_skills,
                                                 n_hidden_filters=self.parameters.n_hidden).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.parameters.obs_shape, n_skills=self.cfg.n_skills,
                                           n_hidden_filters=self.parameters.n_hidden).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.parameters.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.parameters.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.parameters.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.parameters.lr)
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.parameters.lr)

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
  
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.replay_buffer.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))
        
        states = torch.cat(batch.state).view(self.parameters.batch_size, self.parameters.obs_shape + self.cfg.n_skills).to(self.device)
      
        zs = torch.cat(batch.z).view(self.parameters.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.parameters.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.parameters.action_shape).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.parameters.batch_size, self.parameters.obs_shape + self.cfg.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.replay_buffer) < self.parameters.batch_size:
            return None
        else:
            batch = self.replay_buffer.sample(self.parameters.batch_size)
           
            states, zs, dones, actions, next_states = self.unpack(batch)
          
            p_z = from_numpy(self.p_z).to(self.device)
           
            # Calculating the value target
            
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2) #Double Q-Learning
            target_value = q.detach() - self.parameters.alpha * log_probs.detach()
            
            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.parameters.obs_shape, self.cfg.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.parameters.reward_scale * rewards.float() + \
                           self.parameters.gamma * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.parameters.alpha * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.parameters.obs_shape, self.cfg.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.parameters.tau * local_param.data +
                                    (1 - self.parameters.tau) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.replay_buffer.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.replay_buffer.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)