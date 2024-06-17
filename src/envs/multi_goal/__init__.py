from gymnasium.envs.registration import register

register(
    id='MultiGoal-v0',
    entry_point='envs.multi_goal.multi_goal:MultiGoal'
)

