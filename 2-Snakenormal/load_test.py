import torch
from model import Linear_QNet
from snake_gameai import SnakeGameAI

# Load a specific model
model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load('./saved_models/best_model_epoch_120_score_53.h5'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Example: Test the loaded model
def test_model():
    game = SnakeGameAI()
    while True:
        # Replace with actual state extraction and action logic
        state = [0] * 11  # Dummy state for demonstration
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = model(state_tensor)
        action = torch.argmax(prediction).item()

        # Perform action in the game
        reward, done, score = game.play_step([0, 1, 0])  # Example action
        if done:
            print(f"Game Over. Score: {score}")
            game.reset()

test_model()
