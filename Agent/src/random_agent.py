import gymnasium as gym

env = gym.make("FrozenLake-v1")

episodes = 100

successful_episodes = 0
total_reward = 0

for episode in range(episodes):

    observation, info = env.reset()

    terminated = False
    truncated = False

    episode_reward = 0

    while not terminated and not truncated:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

    total_reward += episode_reward

    if episode_reward > 0:
        successful_episodes += 1

success_rate = successful_episodes / episodes
average_reward = total_reward / episodes

print("Episodes:", episodes)
print("Successful episodes:", successful_episodes)
print("Success rate:", success_rate)
print("Average reward:", average_reward)

env.close()