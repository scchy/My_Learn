# python3
# Create date: 2023-11-29
# Func: test mario
# ===============================================================

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = JoypadSpace(env, [['right'], ['right', 'A']])

state = env.reset()
print(state.shape)
for step in range(5000):
    env.render()
    state, reward, done, info, _ = env.step(env.action_space.sample())
    if done:
        state = env.reset()

env.close()
