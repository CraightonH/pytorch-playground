import pygame
import numpy as np
import snake
import nn
import sys

def play_snake(episodes=10, max_steps=500, render=True, model_file="snake_dqn.pth"):
    """
    Run the trained Snake agent without training, just playing.
    
    Args:
        episodes (int): Number of episodes to play.
        max_steps (int): Max steps per episode.
        render (bool): Whether to render the game (default True).
        model_file (str): Path to the saved model file.
    """
    # Initialize game and agent
    game = snake.SnakeGame(render=render)
    agent = nn.DQNAgent(state_size=17, action_size=3, load_model=True)  # Load the trained model
    
    # Ensure epsilon is 0 (greedy policy)
    agent.epsilon = 0.0  # No exploration, just exploitation
    
    print(f"Playing {episodes} episodes with trained model from {model_file}")

    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        score = 0
        
        for time in range(max_steps):
            # Get action from the trained model (greedy)
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            
            state = next_state
            total_reward += reward
            score = game.score  # Track score for display
            
            if render:
                game.render()
                # Handle window close
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
            if done:
                break
        
        print(f"Episode: {e+1}/{episodes}, Score: {score}, Total Reward: {total_reward}, "
              f"Final Length: {game.final_length}, Wall Collisions: {game.wall_collisions}, "
              f"Self Collisions: {game.self_collisions}")
    
    if render:
        pygame.quit()

if __name__ == "__main__":
    play_snake(episodes=25, max_steps=500, render=True, model_file="snake_dqn.pth")
