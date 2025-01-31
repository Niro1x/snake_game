import torch
from model import Linear_QNet
from snake_gameai import SnakeGameAI, Point, Direction, BLOCK_SIZE

# Load the trained model
model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load('./saved_models/best_model_epoch_120_score_53.h5'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Function to extract the state (same as training)
def get_state(game):
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
    return torch.tensor(state, dtype=torch.float)

# Test the model
def test_model():
    game = SnakeGameAI()
    while True:
        # Get the current state
        state = get_state(game).unsqueeze(0)

        # Get the model's action prediction
        prediction = model(state)
        action = torch.argmax(prediction).item()

        # Perform the action in the game
        final_action = [0, 0, 0]
        final_action[action] = 1
        reward, done, score = game.play_step(final_action)

        if done:
            print(f"Game Over. Score: {score}")
            game.reset()

if __name__ == "__main__":
    test_model()
