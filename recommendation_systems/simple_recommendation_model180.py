import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(SimpleRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        return (user_vec * item_vec).sum(dim=1)

class SampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(data, num_users, num_items, embedding_dim=8, epochs=5, batch_size=32):
    train_data, val_data = train_test_split(data, test_size=0.2)
    train_dataset = SampleDataset(train_data)
    val_dataset = SampleDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleRecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        for user_id, item_id, rating in train_loader:
            optimizer.zero_grad()
            output = model(user_id, item_id)
            loss = criterion(output, rating.float())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return model

if __name__ == '__main__':
    mock_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    num_users = 2
    num_items = 2
    model = train_model(mock_data, num_users, num_items)
    print('Model trained successfully!')