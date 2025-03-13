import csv
import os
from collections import deque
import pygame
import numpy as np
import snake
import nn
import sys

def train_snake(load_model=False, render=False, metrics_file="metrics/snake_metrics_dqn.csv"):
    game = snake.SnakeGame(render=render)
    agent = nn.DQNAgent(state_size=13, action_size=3, load_model=load_model)
    episodes = 2000
    max_steps = 500
    batch_size = 1024

    scores = deque(maxlen=100)
    rewards = deque(maxlen=100)
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, 'a', newline='') as csvfile:
        fieldnames = [
            'Episode', 'Score', 'Reward', 'Epsilon',
            'Avg_Score_50', 'Avg_Score_100', 'Avg_Reward_50', 'Avg_Reward_100',
            'Self_Collisions', 'Wall_Collisions', 'Final_Length'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        executed_episodes = 0
        
        try:
            for e in range(episodes):
                state = game.reset()
                total_reward = 0
                for time in range(max_steps):  # Max steps per episode
                    action = agent.act(state)
                    next_state, reward, done = game.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    
                    if render:
                        game.render()  # Visualize the game
                        # print(f"Episode {e+1}, Step {time+1}", end='\r')
                        sys.stdout.flush()

                    agent.replay(batch_size)

                    if done:
                        break

                executed_episodes += 1
                scores.append(game.score)
                rewards.append(total_reward)

                # Calculate running averages
                avg_score_50 = np.mean(list(scores)[-50:]) if len(scores) >= 50 else np.mean(scores)
                avg_score_100 = np.mean(scores) if len(scores) > 0 else 0
                avg_reward_50 = np.mean(list(rewards)[-50:]) if len(rewards) >= 50 else np.mean(rewards)
                avg_reward_100 = np.mean(rewards) if len(rewards) > 0 else 0                  
                print(f"Episode: {e+1}/{episodes}, "
                        f"Score: {game.score}, "
                        f"Reward: {total_reward:.2f}, "
                        f"Epsilon: {agent.epsilon:.2f}, "
                        f"Avg Score (50): {avg_score_50:.2f}, "
                        f"Avg Score (100): {avg_score_100:.2f}, "
                        f"Avg Reward (50): {avg_reward_50:.2f}, "
                        f"Avg Reward (100): {avg_reward_100:.2f}, "
                        f"Self Collisions: {game.self_collisions}, "
                        f"Wall Collisions: {game.wall_collisions}, "
                        f"Final Length: {game.final_length}")
                
                sys.stdout.flush()

                # Write metrics to CSV
                writer.writerow({
                    'Episode': e + 1,
                    'Score': game.score,
                    'Reward': total_reward,
                    'Epsilon': agent.epsilon,
                    'Avg_Score_50': avg_score_50,
                    'Avg_Score_100': avg_score_100,
                    'Avg_Reward_50': avg_reward_50,
                    'Avg_Reward_100': avg_reward_100,
                    'Self_Collisions': game.self_collisions,
                    'Wall_Collisions': game.wall_collisions,
                    'Final_Length': game.final_length
                })
                csvfile.flush()  # Ensure data is written immediately

                if render:  # Only check events if rendering
                    pygame.event.clear()  # Clear event queue
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            # agent.save_model()
                            pygame.quit()
                            return
                    pygame.display.update()

            # Save model after training completes
            print(f"Completed {executed_episodes} episodes")
            agent.save_model()

        except KeyboardInterrupt:
            # Save if interrupted (e.g., Ctrl+C)
            agent.save_model()
            pygame.quit()
            print(f"Interrupted after {executed_episodes} episodes, model saved")
            return

if __name__ == "__main__":
    train_snake(load_model=True, render=True)
