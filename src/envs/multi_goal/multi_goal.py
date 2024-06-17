import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pygame
import os

import gymnasium as gym
from gymnasium import spaces

class MultiGoal(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, goal_reward=10, actuation_cost_coeff=30,
                 distance_cost_coeff=1, init_sigma=0.1, render_mode = "rgb_array"):
        
        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0, 0), dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )
        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        #self.reset()
        self.observation = None

        self._ax = None
        self._env_lines = list()
        self.fixed_plots = None
        self.dynamic_plots = []
        
        #self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        self.observation_space = spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None
        )
        
        self.action_space = spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,)
        )
      
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = float(self.observation_space.low[0]), float(self.observation_space.high[0])
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        
        #info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return [self.observation], {}
    
    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = float(self.action_space.low[0]), float(self.action_space.high[0])
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb, o_ub = float(self.observation_space.low[0]), float(self.observation_space.high[0])
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        reward = self.compute_reward(self.observation, action)
        cur_position = self.observation
        dist_to_goal = np.amin([
            np.linalg.norm(cur_position - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        if self.render_mode == "human":
            self._render_frame()

        return [self.observation], reward, done, False, {'pos': next_obs}
    
    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, paths = []):
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = list()

        for path in paths:
            path = np.array([p[0] for p in path])
        
            xx = path[:, 0]
            yy = path[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
    
    def save_environment_plot(self, folder, filename, paths = []):
        os.makedirs(folder, exist_ok=True)
        
        self._render_frame(paths)
        plt.savefig(os.path.join(folder, filename))
        
    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]
    
    def _init_plot(self):
        self.fig_env = plt.figure(figsize=(7, 7))
        self._ax = self.fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next