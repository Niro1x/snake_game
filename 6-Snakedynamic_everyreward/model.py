import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='best_model.h5'):
        # Define the save directory
        model_folder_path = './saved_models'
        
        # Create the directory if it doesn't exist
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Full file path
        file_name = os.path.join(model_folder_path, file_name)
        
        # Save the model weights
        torch.save(self.state_dict(), file_name)
        print(f"Best Model Saved to {file_name}")

    def load(self, file_name='best_model.h5'):
        # Define the load directory
        model_folder_path = './saved_models'
        file_name = os.path.join(model_folder_path, file_name)
        
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print(f"Model loaded from {file_name}")
        else:
            raise FileNotFoundError(f"Model file {file_name} not found.")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.model.linear1.weight.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.model.linear1.weight.device)
        action = torch.tensor(action, dtype=torch.long).to(self.model.linear1.weight.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.model.linear1.weight.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Predicted Q values
        pred = self.model(state)

        # 2. Q_new = reward + gamma * max(next_predicted Q value)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
