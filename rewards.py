### Wall Collision
WALL_COLLISION_CONSTANT = -10
WALL_COLLISION_MULTIPLIER = 1.2

### Self Collision
SELF_COLLISION_CONSTANT = -10
SELF_COLLISION_MULTIPLIER = 2

### Eats Food
FOOD_EATEN_CONSTANT = 20
FOOD_EATEN_MULTIPLIER = 1.5

### Approaches Food
FOOD_PATH_CONSTANT = 1
FOOD_PATH_MULTIPLIER = 1

### Loops
LOOP_CONSTANT = -5
LOOP_MULTIPLIER = 1

### Wanders
WANDER_CONSTANT = -1
WANDER_MULTIPLIER = .1

def calculate_reward(new_head, food, snake, last_positions, wall_collision, self_collision, approaching_food):
    """
    Calculate the reward based on the game state.
    
    Args:
        new_head (tuple): New head position after move (x, y).
        head (tuple): Current head position before move (x, y).
        food (tuple): Food position (x, y).
        snake (list): List of snake segments [(x1, y1), ...].
        last_positions (deque): Last 4 head positions to detect spinning.
        wall_collision (bool): True if new_head hits a wall.
        self_collision (bool): True if new_head hits the snake body.
        approaching_food (bool): True if move gets closer to food.
    
    reward =s:
        float: Reward value for the action.
    """
    reward = 0
    if wall_collision:
        reward = WALL_COLLISION_CONSTANT - (WALL_COLLISION_MULTIPLIER * len(snake))
    elif self_collision:
        reward = SELF_COLLISION_CONSTANT - (SELF_COLLISION_MULTIPLIER * len(snake))
    else:
        if new_head == food:
            reward = FOOD_EATEN_CONSTANT + (FOOD_EATEN_MULTIPLIER * len(snake))
        elif approaching_food:
            reward = FOOD_PATH_CONSTANT * FOOD_PATH_MULTIPLIER
        else:
            if len(last_positions) == 4 and len(set(last_positions)) < 3:
                reward = LOOP_CONSTANT * LOOP_MULTIPLIER # -5 * len(self.snake) # disincentize looping when snake grows # Stuck in a loop
            else:
                reward = WANDER_CONSTANT + (WANDER_MULTIPLIER * min(len(snake) - 1, 10))  # disincentivize wandering early
    # print(f"reward: {reward:.2f}")
    return reward
