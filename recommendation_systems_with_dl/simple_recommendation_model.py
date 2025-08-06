import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(SimpleRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec * item_vec).sum(1)

class MockDataset(Dataset):
    def __init__(self, user_item_pairs, ratings):
        self.user_item_pairs = user_item_pairs
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_item_pairs[idx], self.ratings[idx]

def train_model(user_item_pairs, ratings, num_users, num_items, embedding_dim=8, epochs=5, batch_size=2):
    model = SimpleRecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = MockDataset(user_item_pairs, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for users, items in dataloader:
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings[users])
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return model

if __name__ == '__main__':
    user_item_pairs = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    ratings = torch.tensor([5.0, 4.0, 3.0, 2.0])
    model = train_model(user_item_pairs[:, 0], ratings, num_users=2, num_items=2)
    print('Model trained successfully.')