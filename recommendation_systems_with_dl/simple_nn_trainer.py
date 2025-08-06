import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_multilabel_classification
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    """A simple feedforward neural network for multi-label classification."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def create_data_loader(num_samples=1000, num_features=20, num_classes=5):
    """Creates a DataLoader for a multi-label classification dataset."""
    X, y = make_multilabel_classification(n_samples=num_samples, n_features=num_features, n_classes=num_classes, n_labels=2, random_state=42)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(num_epochs=10):
    """Trains the SimpleNN model with a sample dataset."""
    input_size = 20
    hidden_size = 10
    output_size = 5
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loader = create_data_loader()

    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train_model()