from stable_baselines3.common.buffers import ReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, input_dim, env, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.action_space.shape))
        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
     
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
class SACAgent:
    def __init__(self,
                 cfg,
                 train_env,
                 p_z = None):
        self.cfg = cfg
        self.train_env = train_env
      
        self.rb = ReplayBuffer(
            cfg.replay_buffer_size,
            self.train_env.observation_space,
            self.train_env.action_space,
            cfg.device,
            handle_timeout_termination=False,
        )
        input_dim = np.prod(cfg.agent.parameters.obs_shape)
        action_dim = np.prod(cfg.agent.parameters.action_shape)
      
        self.actor = Actor(env = train_env, input_dim = input_dim, hidden_dim = self.cfg.agent.parameters.hidden_dim_actor).to(cfg.device)
        self.qf1 = SoftQNetwork(input_dim = input_dim, action_dim = action_dim, hidden_dim=self.cfg.agent.parameters.hidden_dim_critic).to(cfg.device)
        self.qf2 = SoftQNetwork(input_dim = input_dim, action_dim = action_dim, hidden_dim=self.cfg.agent.parameters.hidden_dim_critic).to(cfg.device)
        self.qf1_target = SoftQNetwork(input_dim = input_dim, action_dim = action_dim, hidden_dim=self.cfg.agent.parameters.hidden_dim_critic).to(cfg.device)
        self.qf2_target = SoftQNetwork(input_dim = input_dim, action_dim = action_dim, hidden_dim=self.cfg.agent.parameters.hidden_dim_critic).to(cfg.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr= cfg.agent.parameters.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr= cfg.agent.parameters.policy_lr)
        
        # Automatic entropy tuning
        if cfg.agent.parameters.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(train_env.action_space.shape).to(cfg.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
            self.alpha = log_alpha.exp().item()
            self.a_optimizer = optim.Adam([log_alpha], lr=cfg.agent.parameters.q_lr)
        else:
            self.alpha = cfg.agent.parameters.alpha
    
    def select_action(self, obs, global_step):
        if global_step < self.cfg.agent.parameters.learning_starts:
            actions = np.array([self.train_env.action_space.sample()])
        else:
            obs = torch.Tensor(obs).to(self.cfg.device)
            actions, _, _ = self.actor.get_action(obs)
            actions = actions.detach().cpu().numpy()
        return actions
                
    def update(self, global_step, obs, next_obs, action, rewards, dones, infos):
        
        self.rb.add(obs, next_obs, action, rewards, dones, infos)
        metrics = None
        # ALGO LOGIC: training.
        if global_step > self.cfg.agent.parameters.learning_starts:
            data = self.rb.sample(self.cfg.agent.parameters.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
              
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.agent.parameters.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if global_step % self.cfg.agent.parameters.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    self.cfg.agent.parameters.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.get_action(data.observations)
                    qf1_pi = self.qf1(data.observations, pi)
                    qf2_pi = self.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.cfg.agent.parameters.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data.observations)
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.cfg.agent.parameters.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
                
                metrics = {
                    "losses/qf1_values" : qf1_a_values.mean().item(),
                    "losses/qf2_values" : qf2_a_values.mean().item(),
                    "losses/qf1_loss" : qf1_loss.item(),
                    "losses/qf2_loss" : qf2_loss.item(),
                    "losses/qf_loss" : qf_loss.item(),
                    "losses/actor_loss" : actor_loss.item(),
                    "losses/alpha" : self.alpha,
                }
                if self.cfg.agent.parameters.autotune:
                    metrics["losses/alpha_loss"] = alpha_loss.item()
                
            # update the target networks
            if global_step % self.cfg.agent.parameters.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.cfg.agent.parameters.tau * param.data + (1 - self.cfg.agent.parameters.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.cfg.agent.parameters.tau * param.data + (1 - self.cfg.agent.parameters.tau) * target_param.data)
            
            
           
            
        return metrics
            
           
                    
      