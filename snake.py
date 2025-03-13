from collections import deque
import numpy as np
import random
import pygame

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
SNAKE_SPEED = 30

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self, render=False):
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        self.last_positions = deque(maxlen=4)
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.self_collisions = 0
        self.wall_collisions = 0
        self.final_length = len(self.snake)  # Will update on end
        self.last_positions.clear()
        self.last_positions.append(self.snake[0])
        if self.render_mode:
            self.render()  # Render initial state
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def approaching_food(self, head, new_head):
        old_distance_to_food_x = abs(head[0] - self.food[0])
        old_distance_to_food_y = abs(head[1] - self.food[1])
        new_distance_to_food_x = abs(new_head[0] - self.food[0])
        new_distance_to_food_y = abs(new_head[1] - self.food[1])
        # print("new ({}, {}), old ({}, {})".format(new_distance_to_food_x, new_distance_to_food_y, old_distance_to_food_x, old_distance_to_food_y))
        if (new_distance_to_food_x < old_distance_to_food_x) or (new_distance_to_food_y < old_distance_to_food_y):
            # print("getting closer")
            return True
        return False

    def step(self, action):
      # Action: 0 = straight, 1 = left, 2 = right
        reward = 0
        if action == 1:  # Turn left
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # Turn right
            self.direction = (self.direction[1], -self.direction[0])

        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check collision with walls or self
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.done = True
            self.wall_collisions += 1
            reward = -5
        elif new_head in self.snake:
            self.done = True
            self.self_collisions += 1
            reward = -5 #- 1.2 * len(self.snake) # potentially reduce penalty to encourage more growth later
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:  # Ate food
                self.score += 1
                self.food = self._place_food()
                reward = 25 * min(self.score + 1, 10)
            elif self.approaching_food(head, new_head):
                reward = 1 # incentivize hunting
                self.snake.pop()  # Remove tail
            else:
                self.snake.pop()  # Remove tail
                if len(self.last_positions) == 4 and len(set(self.last_positions)) < 3:
                    reward = -2 # Stuck in a loop
                else:
                    reward = -1 # disincentivize wandering

        if self.done:
            self.final_length = len(self.snake)

        return self._get_state(), reward, self.done

    def _get_state(self):
        head_x, head_y = self.snake[0]
        tail_x, tail_y = self.snake[-1]
        food_x, food_y = self.food
        food_dx = head_x - food_x 
        food_dy = head_y - food_y 
        tail_dx = tail_x - head_x
        tail_dy = tail_y - head_y
        
        # Wall distances
        wall_left = 1 if head_x <= 2 else 0
        wall_right = 1 if head_x >= GRID_WIDTH - 3 else 0
        wall_up = 1 if head_y <= 2 else 0
        wall_down = 1 if head_y >= GRID_HEIGHT - 3 else 0

        # print(f"food_dx: {food_dx}, food_dy: {food_dy}, facing_dir: {facing_dir}, head_x: {head_x}, head_y: {head_y}, snake_length: {len(self.snake)}, wall_left: {wall_left}, wall_right: {wall_right}, wall_up: {wall_up}, wall_down: {wall_down}")

        return np.array([
            food_dx,
            food_dy,
            self.direction[0],
            self.direction[1],
            head_x,
            head_y,
            len(self.snake),
            wall_left,
            wall_right,
            wall_up,
            wall_down,
            tail_dx,
            tail_dy
        ], dtype=np.float32)

    def render(self):
        if not self.render_mode:
            return  # Skip rendering if not enabled
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.event.pump()  # Process events before flip
        pygame.display.flip()
        self.clock.tick(SNAKE_SPEED)
        pygame.time.wait(10)  # 10ms delay
