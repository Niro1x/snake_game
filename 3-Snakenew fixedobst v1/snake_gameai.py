import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

# Define constants
BLOCK_SIZE = 20
SPEED = 0  # Set to 0 to run the game as fast as possible during training
WHITE = (255, 255, 255)
RED = (200, 0, 0)  # Food color
BLUE1 = (0, 0, 255)  # Snake color
BLUE2 = (0, 100, 255)  # Snake inner color
BLACK = (0, 0, 0)  # Background color
GRAY = (128, 128, 128)  # Obstacle color


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x , y")


class SnakeGameAI:
    def __init__(self, w=640, h=480, render=False):
        self.w = w
        self.h = h
        self.render = render
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self.obstacles = self._create_fixed_obstacles()  # Create fixed obstacles
        self._place_food()
        self.frame_iteration = 0

    def _create_fixed_obstacles(self):
        obstacles = []

        # Obstacle 1: Single point
        obstacles.append(Point(100, 100))

        # Obstacle 2: Horizontal line (5 blocks)
        for i in range(5):
            obstacles.append(Point(200 + i * BLOCK_SIZE, 300))

        # Obstacle 3: Vertical line (5 blocks)
        for i in range(5):
            obstacles.append(Point(400, 200 + i * BLOCK_SIZE))

        # Obstacle 4: L-shaped obstacle
        # Horizontal part
        for i in range(3):
            obstacles.append(Point(500 + i * BLOCK_SIZE, 100))
        # Vertical part
        for i in range(1, 4):
            obstacles.append(Point(500, 100 + i * BLOCK_SIZE))

        # Obstacle 5: Square shape (2x2 blocks)
        obstacles.append(Point(150, 400))
        obstacles.append(Point(150 + BLOCK_SIZE, 400))
        obstacles.append(Point(150, 400 + BLOCK_SIZE))
        obstacles.append(Point(150 + BLOCK_SIZE, 400 + BLOCK_SIZE))

        return obstacles

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                break

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        self._update_ui()
        if self.render:
            self.clock.tick(SPEED)
        # 6. Return game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        if not self.render:
            return
        self.display.fill(BLACK)
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.display, GRAY, pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
            )
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )
        # Draw food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right turn, left turn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        action_idx = np.argmax(action)
        if action_idx == 0:  # Straight
            new_dir = clock_wise[idx]
        elif action_idx == 1:  # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        # Hits an obstacle
        if pt in self.obstacles:
            return True
        return False
