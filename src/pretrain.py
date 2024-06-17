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
from experiment.experiment import Experiment


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.1")
def main(cfg: DictConfig) -> None:
    experiment = Experiment(cfg)
    root_dir = Path.cwd()
    #snapshot = root_dir / 'snapshot.pt'
    #if snapshot.exists():
    #    print(f'resuming: {snapshot}')
    #    experiment.load_snapshot()
    experiment.train()
   
if __name__ == "__main__":
    main()