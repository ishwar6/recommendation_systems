import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    """Custom dataset for reinforcement learning with mock data."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SimplePolicyNetwork(nn.Module):
    """A simple policy network for reinforcement learning."""
    def __init__(self, input_dim, output_dim):
        super(SimplePolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def train_policy_network(data, input_dim, output_dim, num_epochs=100):
    """Train the policy network using the provided data."""
    dataset = SimpleDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimplePolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for states, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(states.float())
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
    return model

if __name__ == '__main__':
    mock_data = (torch.randn(100, 4), torch.randint(0, 2, (100,)))
    trained_model = train_policy_network(mock_data, input_dim=4, output_dim=2)
    print(trained_model)