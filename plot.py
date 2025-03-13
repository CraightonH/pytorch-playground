import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
data = pd.read_csv("metrics/snake_metrics_dqn_w_walls_tail.csv")

# Create subplots
plt.figure(figsize=(12, 12))

# Scores
plt.subplot(3, 1, 1)
plt.plot(data['Episode'], data['Score'], label='Score', alpha=0.5)
plt.plot(data['Episode'], data['Avg_Score_50'], label='Avg Score (50)')
plt.plot(data['Episode'], data['Avg_Score_100'], label='Avg Score (100)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Snake Game Scores Over Time')
plt.legend()
plt.grid(True)

# Rewards
plt.subplot(3, 1, 2)
plt.plot(data['Episode'], data['Reward'], label='Reward', alpha=0.5)
plt.plot(data['Episode'], data['Avg_Reward_50'], label='Avg Reward (50)')
plt.plot(data['Episode'], data['Avg_Reward_100'], label='Avg Reward (100)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Snake Game Rewards Over Time')
plt.legend()
plt.grid(True)

# New Metrics
plt.subplot(3, 1, 3)
plt.plot(data['Episode'], data['Self_Collisions'], label='Self Collisions', alpha=0.5)
plt.plot(data['Episode'], data['Wall_Collisions'], label='Wall Collisions', alpha=0.5)
plt.plot(data['Episode'], data['Final_Length'], label='Final Length')
plt.xlabel('Episode')
plt.ylabel('Count / Length')
plt.title('Collisions and Snake Length Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()