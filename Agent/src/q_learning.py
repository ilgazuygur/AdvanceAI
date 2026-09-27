import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")

num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-table starts with all values at 0
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
episodes = 5000
alpha = 0.8
gamma = 0.95
epsilon = 1.0

epsilon_decay = 0.995
min_epsilon = 0.01

successful_episodes = 0

for episode in range(episodes):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Best Q-value in the next state
        best_next_value = np.max(q_table[next_state])

        # Q-Learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward
            + gamma * best_next_value
            - q_table[state, action]
        )

        state = next_state

        if reward == 1:
            successful_episodes += 1

    # Slowly reduce exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


print("Training finished.")
print("Successful training episodes:", successful_episodes)

print("\nLearned Q-table:")
print(q_table)

# Test the learned agent

test_episodes = 100
test_successes = 0

for episode in range(test_episodes):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:

        # No exploration during testing
        action = np.argmax(q_table[state])

        state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            test_successes += 1

test_success_rate = test_successes / test_episodes

print("\nTesting results:")
print("Successful episodes:", test_successes)
print("Success rate:", test_success_rate)

# Save the learned Q-table
np.save("results/q_table.npy", q_table)
print("Q-table saved.")


env.close()