import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
#os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import wandb

import utils
from experiment.logger import Logger
#from envs.dmc_benchmark import PRIMAL_TASKS
#import envs.dmc as dmc
#from dm_env import specs
#from utils.replay_buffer_old import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder
import numpy as np
from tqdm import tqdm

import envs

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def create_env(cfg):
    if cfg.env.id in ["MultiGoal"]:
        import gymnasium as gym
        train_env = gym.make(cfg.env.id)
        eval_env = gym.make(cfg.env.id)
    """
    else:
        task = PRIMAL_TASKS[cfg.domain]

        train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed)
        eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed)
    """
    return train_env, eval_env
    
def make_agent(train_env, cfg, p_z = None):
    
    #Update CFG
    cfg.agent.parameters.obs_type = cfg.obs_type
    if cfg.env.id in ["MultiGoal"]:
        cfg.agent.parameters.obs_shape = train_env.observation_space.shape
        cfg.agent.parameters.action_shape = train_env.action_space.shape
    else:
        cfg.agent.parameters.obs_shape = train_env.observation_spec().shape
        cfg.agent.parameters.action_shape = train_env.action_spec().shape
    #cfg.agent.parameters.num_expl_steps = num_expl_steps
    #cfg.agent.parameters.action_bounds = [float(train_env.action_space.low[0]), float(train_env.action_space.high[0])]
    
    #Load agent
    if cfg.agent.name == "sac":
        from agents.sac import SACAgent
        agent = SACAgent(cfg = cfg, p_z = p_z, train_env=train_env)
    elif cfg.agent.name == "diayn":
        from agents.diayn import DIAYNAgent
        agent = DIAYNAgent(cfg = cfg, train_env = train_env)
    return agent

class Experiment:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.env.id, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project=cfg.wandb_project, group=cfg.agent.name, name=exp_name)
            
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb,
                             prefix_filename="train")
    
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,
            camera_id=0,
            use_wandb=self.cfg.use_wandb)
        
        #self.train_env, self.test_env = create_env(cfg)
    
    def train(self):
        
        self.train_env, self.test_env = create_env(self.cfg)
      
        self.agent = make_agent(self.train_env,
                                self.cfg)

        obs, _ = self.train_env.reset(seed=self.cfg.seed)
        #self.train_video_recorder.init(self.train_env.render())
   
        for episode in tqdm(range(self.cfg.n_episodes)):
            
            self.agent.initialize()
            episode_return, episode_discounted_return, episode_length = 0.0, 0.0, 0
            #trajectory = []
            for t in range(self.cfg.max_episode_length):
                step = t + episode * self.cfg.max_episode_length

                action = self.agent.select_action(obs, episode)
         
                next_obs, reward, done, truncate, info = self.train_env.step(action)
                episode_return += reward
                episode_length += 1
                episode_discounted_return += reward * self.cfg.agent.parameters.gamma
      
                metrics = self.agent.update(episode, obs, next_obs, action, reward, done, info)
                #trajectory.append(obs)
                
                if step % self.cfg.log_frequency == 0:
                    if metrics is not None:
                        self.logger.log_metrics(metrics, step, "step")
                        self.logger.dump(step, "step")
  
                obs = next_obs
                
                if done or truncate:
                    obs, _ = self.train_env.reset(seed=self.cfg.seed)
                    #policy.reset()
                    self.logger.log_metrics({
                        "episode_return" : episode_return,
                        "episode_discounted_return" : episode_discounted_return,
                        "episode_length" : episode_length
                        }, episode, "episode")
                    self.logger.dump(episode, "episode")
                    episode_return, episode_discounted_return, episode_length = 0.0, 0.0, 0
                    #self.train_video_recorder.save(f'{step}.mp4')
                    
                    #self.train_env.save_environment_plot(self.work_dir, 'trajectory_%d'  % (step), paths = [trajectory])
                    
                    break
                
     
                
           
