import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(
        self, input_size, hidden_size1, hidden_size2, output_size, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size1).to(self.device)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2).to(self.device)
        self.linear3 = nn.Linear(hidden_size2, output_size).to(self.device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name="best_model.pth"):
        model_folder_path = "./saved_models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load(self, file_name="best_model.pth"):
        model_folder_path = "./saved_models"
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name, map_location=self.device))
            print(f"Model loaded from {file_name}")
        else:
            raise FileNotFoundError(f"Model file {file_name} not found.")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        device = self.model.device  # Ensure we're using the model's device
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

        # Predicted Q values with current state
        pred = self.model(state)

        # Predicted Q values with next state
        with torch.no_grad():
            next_pred = self.model(next_state)

        # Calculate target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_pred[idx])
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Compute loss and optimize
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
