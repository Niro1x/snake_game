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
BATCH_SIZE = 2048
LR = 0.0005
UPDATE_TARGET = 5  # Update target network every 5 games

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
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay memory
        self.model = Linear_QNet(12, 256, 256, 3, device=device)
        self.target_model = Linear_QNet(12, 256, 256, 3, device=device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

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

        # Danger in all directions
        danger_straight = game.is_collision(
            point_r if dir_r else point_l if dir_l else point_u if dir_u else point_d
        )
        danger_right = game.is_collision(
            point_d if dir_r else point_u if dir_l else point_r if dir_u else point_l
        )
        danger_left = game.is_collision(
            point_u if dir_r else point_d if dir_l else point_l if dir_u else point_r
        )

        # Food relative position
        dist_to_food_x = (game.food.x - head.x) / game.w
        dist_to_food_y = (game.food.y - head.y) / game.h

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
            # Food relative position
            dist_to_food_x,
            dist_to_food_y,
            # Tail proximity (placeholder)
            0,
            # Obstacle proximity (placeholder)
            0,
            # Bias term (optional)
            1,
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
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
            state0 = torch.tensor(state, dtype=torch.float).to(device)
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
    game = SnakeGameAI(render=True)

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

            # Update target model
            if agent.n_game % UPDATE_TARGET == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

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
            if agent.n_game >= 5000:
                print("Training completed after 5000 games.")
                print("Final Highest Record:", record)
                break


if __name__ == "__main__":
    train()
