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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions for saving/loading record
def save_record(record, file_name='record.txt'):
    with open(file_name, 'w') as f:
        f.write(str(record))

def load_record(file_name='record.txt'):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            return int(f.read())
    return 0

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)  # Move model to device
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

        state = [
            # Danger straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY exceeded

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Sample a batch
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration/exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)  # Move state to device
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = load_record()  # Load previous high score
    last_20_scores = []  # Track scores for the last 20 games
    agent = Agent()
    game = SnakeGameAI()

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

            # Append score to the last 20 scores list
            last_20_scores.append(score)
            if len(last_20_scores) > 20:
                last_20_scores.pop(0)  # Maintain only the last 20 scores

            # Every 20 epochs, calculate the best score of the last 20 games
            if agent.n_game % 20 == 0:
                best_last_20_score = max(last_20_scores)  # Best score from last 20 games
                print(f"Best score of the last 20 games: {best_last_20_score}")

                if best_last_20_score > record:
                    record = best_last_20_score
                    save_record(record)  # Save the new record persistently

                    # Save the best model with epoch and score appended to filename
                    filename = f"best_model_epoch_{agent.n_game}_score_{record}.h5"
                    agent.model.save(filename)
                    print(f"New Best Model Saved: {filename}")
                else:
                    print(f"No new best score at Epoch {agent.n_game}. Highest Record: {record}")

            print(f"Game: {agent.n_game}, Score: {score}, Highest Record: {record}")

            # Update plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Stop training after 1000 epochs
            if agent.n_game >= 1000:
                print("Training completed after 1000 games.")
                print("Final Highest Record:", record)
                break



if __name__ == "__main__":
    train()
