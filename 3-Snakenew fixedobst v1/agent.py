import torch
import random
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from Helper import plot
import os

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper functions for saving/loading record
def save_record(record, file_name="record.txt"):
    with open(file_name, "w") as f:
        f.write(str(record))


def load_record(file_name="record.txt"):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return int(f.read())
    return 0


class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 80  # Initial epsilon value
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.epsilon_decay = 0.995  # Decay rate
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # deque for replay memory
        self.model = Linear_QNet(
            11, 256, 256, 3, device=device
        )  # Input size adjusted if necessary
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Danger straight
        danger_straight = (
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d))
        )

        # Danger right
        danger_right = (
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d))
        )

        # Danger left
        danger_left = (
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d))
        )

        state = [
            # Danger straight
            danger_straight,
            # Danger right
            danger_right,
            # Danger left
            danger_left,
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y,  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def get_action(self, state):
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        final_move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(
                device
            )  # Move state to device
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0)
            self.model.train()
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = load_record()  # Load previous high score
    agent = Agent()
    game = SnakeGameAI(render=True)  # Set render=True to visualize

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the experience
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot results
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()

            # Record and save the model if a new high score is achieved
            if score > record:
                record = score
                save_record(record)
                agent.model.save()
                print(f"New High Score! Model saved with score {record}")

            print(f"Game: {agent.n_game}, Score: {score}, Highest Record: {record}")

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Optional: Stop training after a certain number of games
            if agent.n_game >= 1000:
                print("Training completed after 1000 games.")
                print("Final Highest Record:", record)
                break


if __name__ == "__main__":
    train()
