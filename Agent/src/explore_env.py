import gymnasium as gym

env = gym.make("FrozenLake-v1")

observation, info = env.reset()

print("Reset output:")
print("Observation:", observation)
print("Info:", info)

action = env.action_space.sample()

print("\nRandom action:", action)

observation, reward, terminated, truncated, info = env.step(action)

print("\nStep output:")
print("Observation:", observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close()