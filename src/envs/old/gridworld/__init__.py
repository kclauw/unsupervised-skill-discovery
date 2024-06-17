from gymnasium.envs.registration import register

register(
    id='Gridworld-v0',
    entry_point='environments.gridworld.gridworld:GridWorld'
)

