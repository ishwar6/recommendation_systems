import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_multilabel_classification

class MultiLabelDataset(Dataset):
    """Custom dataset for multi-label classification."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNN(nn.Module):
    """Simple neural network for multi-label classification."""
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    """Train the neural network model."""
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=2, random_state=42)
    dataset = MultiLabelDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNN(input_dim=20, output_dim=5)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, epochs=10)
    print('Training complete.')