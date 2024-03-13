import gymnasium as gym
from gymnasium.envs.registration import register
from  ldo_env import LDOEnv

# we register our env here

env_id = 'ldo-v0'

register(
        id = env_id,
        entry_point = 'ldo_env:LDOEnv',
        max_episode_steps = 10,
)
env = gym.make(env_id) 
print("Register the environment success")


observation, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
