import numpy as np
import random
import pygame

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
SNAKE_SPEED = 15

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
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
            reward = -10
        elif new_head in self.snake:
            self.done = True
            self.self_collisions += 1
            reward = -20 - 2 * len(self.snake)
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:  # Ate food
                self.score += 1
                self.food = self._place_food()
                reward = 50
            elif self.approaching_food(head, new_head):
                reward = 1
                self.snake.pop()  # Remove tail
            else:
                self.snake.pop()  # Remove tail
                reward = -1
        if self.done:
            self.final_length = len(self.snake)

        return self._get_state(), reward, self.done

    def _get_state(self):
        head = self.snake[0]
        state = [
            head[0] - self.food[0],  # Distance to food x
            head[1] - self.food[1],  # Distance to food y
            self.direction[0],       # Current direction x
            self.direction[1],       # Current direction y
            head[0],                 # Head x (wall proximity)
            head[1],                 # Head y (wall proximity)
            len(self.snake)          # Snake length
        ]
        return np.array(state, dtype=np.float32)

    def render(self):
        self.screen.fill((0, 0, 0))  # Black background
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.flip()
        self.clock.tick(SNAKE_SPEED)
