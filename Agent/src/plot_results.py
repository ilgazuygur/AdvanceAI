import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results/results.csv")

# Separate training experiments from the random baseline
experiments = data[data["Experiment"] != "Random Agent"].copy()
experiments["Episodes"] = pd.to_numeric(experiments["Episodes"])
experiments = experiments.sort_values("Episodes")

random_rate = data.loc[
    data["Experiment"] == "Random Agent", "Success Rate"
].iloc[0]

# Plot the trained agents
plt.plot(
    experiments["Episodes"],
    experiments["Success Rate"],
    marker="o",
    label="Q-Learning"
)

# Show random performance for comparison
plt.axhline(
    y=random_rate,
    color="red",
    linestyle="--",
    label="Random Agent"
)

plt.xlabel("Training Episodes")
plt.ylabel("Success Rate")
plt.title("Q-Learning Performance on FrozenLake")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("results/learning_curve.png")
plt.close()

print("Graph saved to results/learning_curve.png")