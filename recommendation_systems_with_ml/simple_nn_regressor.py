import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(X, y, input_size, hidden_size, output_size, epochs=100, learning_rate=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.rand(100, 1).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(torch.from_numpy(X_train), torch.from_numpy(y_train), input_size=10, hidden_size=5, output_size=1)
    model.eval()
    with torch.no_grad():
        test_output = model(torch.from_numpy(X_test))
        print('Test Outputs:', test_output.numpy())

if __name__ == '__main__':
    main()