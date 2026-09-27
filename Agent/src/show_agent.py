import gymnasium as gym
import numpy as np
from PIL import Image

# Load what the agent learned
q_table = np.load("results/q_table.npy")

# Get game screens as images
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

try:
    for attempt in range(1, 21):
        state, info = env.reset()
        terminated = False
        truncated = False

        frames = [Image.fromarray(env.render())]

        while not terminated and not truncated:
            # Choose the best learned action
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, info = env.step(action)
            frames.append(Image.fromarray(env.render()))

        if reward == 1:
            frames[0].save(
                "results/learned_agent.gif",
                save_all=True,
                append_images=frames[1:],
                duration=500,
                loop=0
            )

            print("Successful attempt:", attempt)
            print("GIF saved to results/learned_agent.gif")
            break
    else:
        print("No successful episode in 20 attempts. Run again.")
finally:
    env.close()