import gym
from gym import spaces
import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()
pygame.font.init()

# Define constants
BROWN = (139, 69, 19)  # Brown color
SNAKE_COLOR_1 = (0, 255, 0)  # Green color for the first snake
SNAKE_COLOR_2 = (0, 0, 255)  # Blue color for the second snake
FRUIT_COLOR = (255, 0, 0)  # Red color for the fruit
HEAD_COLOR_1 = (0, 128, 0)  # Dark green for the head of the first snake
HEAD_COLOR_2 = (0, 0, 128)  # Dark blue for the head of the second snake

# Custom Gym Environment
class SnakeMultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=500, height=500, rows=7, cols=7):
        super(SnakeMultiAgentEnv, self).__init__()
        
        # Set dimensions
        self.WIDTH = width
        self.HEIGHT = height
        self.ROWS = rows
        self.COLS = cols
        self.CELL_SIZE = self.WIDTH // self.COLS  # Size of each cell
        
        # Define action and observation space for two agents
        self.action_space = spaces.MultiDiscrete([4, 4])  # 4 actions for each snake ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.observation_space = spaces.Box(low=0, high=7, shape=(self.ROWS, self.COLS), dtype=np.uint8)
        
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake Multi-Agent Environment")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake1_positions = self.generate_snake()
        self.snake2_positions = self.generate_snake(exclude=self.snake1_positions)
        self.fruit_position = self.generate_fruit(self.snake1_positions + self.snake2_positions)
        self.directions = ['RIGHT', 'LEFT']  # Initial directions for snake 1 and snake 2
        self.score1 = 0
        self.score2 = 0
        self.steps_without_fruit = 0  # Counter for steps without eating fruit
        
        return self._get_observation()

    def step(self, actions):
        action1, action2 = actions
        
        # Translate actions to directions
        if action1 == 0 and self.directions[0] != 'DOWN':
            self.directions[0] = 'UP'
        elif action1 == 1 and self.directions[0] != 'UP':
            self.directions[0] = 'DOWN'
        elif action1 == 2 and self.directions[0] != 'RIGHT':
            self.directions[0] = 'LEFT'
        elif action1 == 3 and self.directions[0] != 'LEFT':
            self.directions[0] = 'RIGHT'

        if action2 == 0 and self.directions[1] != 'DOWN':
            self.directions[1] = 'UP'
        elif action2 == 1 and self.directions[1] != 'UP':
            self.directions[1] = 'DOWN'
        elif action2 == 2 and self.directions[1] != 'RIGHT':
            self.directions[1] = 'LEFT'
        elif action2 == 3 and self.directions[1] != 'LEFT':
            self.directions[1] = 'RIGHT'

        # Move both snakes
        snake1_positions, self.fruit_position, points1, game_over1, self_collision1 = self.move_snake(
            self.snake1_positions, self.directions[0], self.fruit_position)
        
        snake2_positions, self.fruit_position, points2, game_over2, self_collision2 = self.move_snake(
            self.snake2_positions, self.directions[1], self.fruit_position)

        self.snake1_positions = snake1_positions
        self.snake2_positions = snake2_positions

        self.score1 += points1
        self.score2 += points2

        reward1 = points1 - 0.01  # Penalty for steps
        reward2 = points2 - 0.01

        # Check collisions between the snakes
        snake_collision = any(pos in self.snake2_positions for pos in self.snake1_positions) or \
                          any(pos in self.snake1_positions for pos in self.snake2_positions)

        if self_collision1 or self_collision2 or game_over1 or game_over2 or snake_collision:
            reward1 = reward2 = -5  # Apply penalty for collision
            done = True
        else:
            done = False

        observation = self._get_observation()

        # Additional penalty for too many steps without eating fruit
        self.steps_without_fruit += 1
        if self.steps_without_fruit > 100:
            reward1 -= 1
            reward2 -= 1
            done = True

        return observation, [reward1, reward2], done, {}

    def render(self, mode='human', close=False):
        self.window.fill((255, 255, 255))
        self.draw_grid()
        self.draw_snake(self.snake1_positions, SNAKE_COLOR_1, HEAD_COLOR_1)
        self.draw_snake(self.snake2_positions, SNAKE_COLOR_2, HEAD_COLOR_2)
        self.draw_fruit(self.fruit_position)
        
        # Display scores
        font = pygame.font.Font(None, 36)
        text1 = font.render(f"Snake 1 Score: {self.score1}", True, (0, 0, 0))
        text2 = font.render(f"Snake 2 Score: {self.score2}", True, (0, 0, 0))
        self.window.blit(text1, (10, 10))
        self.window.blit(text2, (10, 50))

        pygame.display.flip()
        self.clock.tick(8)  # Increase frame rate for smoother gameplay

    def close(self):
        pygame.quit()

    def draw_grid(self):
        for row in range(self.ROWS):
            for col in range(self.COLS):
                rect = pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.window, BROWN, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  # Draw cell borders

    def draw_snake(self, snake_positions, snake_color, head_color):
        for i, pos in enumerate(snake_positions):
            rect = pygame.Rect(pos[1] * self.CELL_SIZE, pos[0] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.window, snake_color, rect)
            if i == 0:  # Draw the head of the snake
                center = (pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2, pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2)
                pygame.draw.circle(self.window, head_color, center, self.CELL_SIZE // 4)

    def draw_fruit(self, fruit_position):
        rect = pygame.Rect(fruit_position[1] * self.CELL_SIZE, fruit_position[0] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.window, FRUIT_COLOR, rect)

    def generate_snake(self, exclude=[]):
        while True:
            is_horizontal = random.choice([True, False])
            if is_horizontal:
                row = random.randint(0, self.ROWS - 1)
                col_start = random.randint(0, self.COLS - 2)
                snake_positions = [(row, col_start), (row, col_start + 1)]
            else:
                col = random.randint(0, self.COLS - 1)
                row_start = random.randint(0, self.ROWS - 2)
                snake_positions = [(row_start, col), (row_start + 1, col)]

            # Ensure snake does not spawn on top of existing snake
            if all(pos not in exclude for pos in snake_positions):
                return snake_positions

    def generate_fruit(self, exclude):
        while True:
            fruit_position = (random.randint(0, self.ROWS - 1), random.randint(0, self.COLS - 1))
            if fruit_position not in exclude:
                return fruit_position

    def move_snake(self, snake_positions, direction, fruit_position):
        head_x, head_y = snake_positions[0]
        if direction == 'UP':
            new_head = (head_x - 1, head_y)
        elif direction == 'DOWN':
            new_head = (head_x + 1, head_y)
        elif direction == 'LEFT':
            new_head = (head_x, head_y - 1)
        elif direction == 'RIGHT':
            new_head = (head_x, head_y + 1)

        self_collision = False
        # Check if the snake's head goes out of bounds
        if not (0 <= new_head[0] < self.ROWS and 0 <= new_head[1] < self.COLS):
            return snake_positions, fruit_position, -5, True, self_collision  # Game over with -5 reward for boundary collision

        # Check if the snake's head overlaps with any part of its body
        if new_head in snake_positions:
            self_collision = True
            return snake_positions, fruit_position, -5, True, self_collision  # Game over with -5 reward for self-collision

        # Check if the snake's head overlaps with the fruit
        if new_head == fruit_position:
            fruit_position = self.generate_fruit(snake_positions)  # Generate new fruit position
            self.steps_without_fruit = 0  # Reset counter
            return [new_head] + snake_positions, fruit_position, 10, False, self_collision  # Return 10 points and grow the snake
        else:
            # Add new head and remove tail, ensuring the snake moves forward
            return [new_head] + snake_positions[:-1], fruit_position, 0, False, self_collision  # No points for moving

    def _get_observation(self):
        # Create a blank grid
        grid = np.full((self.ROWS, self.COLS), 0, dtype=np.uint8)
        
        # Set snake 1 position in the grid
        for pos in self.snake1_positions[1:]:
            grid[pos[0], pos[1]] = 1  # 1 for snake 1 body
        head1_pos = self.snake1_positions[0]
        grid[head1_pos[0], head1_pos[1]] = 7  # 7 for snake 1 head

        # Set snake 2 position in the grid
        for pos in self.snake2_positions[1:]:
            grid[pos[0], pos[1]] = 2  # 2 for snake 2 body
        head2_pos = self.snake2_positions[0]
        grid[head2_pos[0], head2_pos[1]] = 6  # 6 for snake 2 head

        # Set fruit position in the grid
        grid[self.fruit_position[0], self.fruit_position[1]] = 3  # 3 for fruit

        return grid


# # To use this environment
if __name__ == "__main__":
    for i in range(20):
        env = SnakeMultiAgentEnv(width=600, height=600, rows=10, cols=10)

        observation = env.reset()
        rewards = [0, 0]

        for step in range(100):
            env.render()
            print(f"Observation: \n {observation}")
            action = env.action_space.sample()  # Take random actions for both snakes
            print(f"Actions: {action}")
            observation, reward, done, info = env.step(action)
            print(f"Observation: \n {observation}")
            print("*****************************")
            rewards = [rewards[0] + reward[0], rewards[1] + reward[1]]
            print(f"{step}: Rewards: {reward}, Cumulative Rewards: {rewards}")
            if done:
                observation = env.reset()
                print(f"Final Cumulative Rewards: {rewards}")
                break

