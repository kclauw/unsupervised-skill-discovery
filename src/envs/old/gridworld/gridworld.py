import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from enum import IntEnum
from gymnasium.core import ActType, ObsType
from collections import Counter
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium import spaces
import os


TILE_MAPPING = {".": 0, "S": 1, "G": 2, "*": 3, "R": 4, "F": 5, "A": 6, "W" : 7}
ACTION_MAPPING = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}

def convert_array_to_minigrid_coords(x, y, height):
    minigrid_y = height - 1 - y
    return (x, minigrid_y)

def map_symbol_to_position(grid_map, symbol):
    pos = np.where(grid_map == symbol)
    x, y = (pos[0].item(), pos[1].item())
    height, width = grid_map.shape
   
    return (y, x)

class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6
    
class GridWorld(MiniGridEnv):
    def __init__(self,
              grid_map_name = None,
              grid_map_folder = "/Users/kenzoclauw/Projects/research/continual-credit-rl/configs/env/gridworld/maps",
              agent_start_pos = None,
              agent_goal_pos = None,
              transition_random_action = 0,
              reward_goal = 1,
              reward_default = 0,
              position_bonus = None,
               max_steps = None,
               **kwargs):
        
        self.grid_map_name = grid_map_name
        self.grid_map_folder = grid_map_folder
        self.grid_map = self.process_gridmap(self.grid_map_name)
        
        self.size = self.grid_map.shape[0]

        self.agent_start_pos = tuple(agent_start_pos) if agent_start_pos else map_symbol_to_position(self.grid_map, 1)
        self.agent_goal_pos = tuple(agent_goal_pos) if agent_goal_pos else map_symbol_to_position(self.grid_map, 2)
        self.agent_start_dir = 0
       
        self.reward_goal = reward_goal
        self.reward_default = reward_default
    
        mission_space = MissionSpace(mission_func=self._gen_mission)
  
        if max_steps is None:
            max_steps = 4 * self.size**2
       
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.n_states = self.width * self.height * 4 #direction
        self.n_actions = self.action_space.n
   
    def set_goal_pos(self, set_goal_pos):
        self.agent_goal_pos = set_goal_pos
         
    def set_start_pos(self, start_pos):
        self.agent_start_pos = start_pos
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        #for i in range(0, height):
        #    self.grid.set(5, i, Wall())
        
        # Place the door and key
        #self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        #self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
    
        self.put_obj(Goal(), self.agent_goal_pos[0], self.agent_goal_pos[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
       
        self.mission = "grand mission"
        print(self.reachable_states())
    
    def reachable_states(self):
        for obj in self.grid.grid:
            print(obj)
        exit(0)
        
        
    @staticmethod
    def _gen_mission():
        return "get to the green goal"

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )
        
        # Check that the agent doesn't overlap with an object
        
        start_cell = self.grid.get(*self.agent_pos)
       
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = self.reward_default
        
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        
        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self.reward_goal
                
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
  
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
      
        return obs, reward, terminated, truncated, {}
    
    def process_gridmap(self, grid_map_name):
        grid_map = open(os.path.join(self.grid_map_folder, grid_map_name + '.txt'), "r").read()
        rows = grid_map.split("\n")
        tiles = [row.split() for row in rows]
        tiles = [[TILE_MAPPING[elem] for elem in row] for row in tiles]
        tiles = np.array(tiles)
        tiles = np.pad(tiles, pad_width=1, mode='constant', constant_values=-1) #Add padding for border around grid
        return tiles
    

