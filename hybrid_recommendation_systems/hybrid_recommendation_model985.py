import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class HybridRecommendationModel(nn.Module):
    """
    A hybrid recommendation model that combines collaborative filtering and content-based filtering.
    """
    def __init__(self, num_users, num_items, embedding_dim):
        super(HybridRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_id, item_id):
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        combined = torch.cat([user_vec, item_vec], dim=1)
        score = self.fc(combined)
        return score.squeeze()

class SampleDataset(Dataset):
    """
    A simple dataset for loading user-item interactions.
    """
    def __init__(self, user_item_pairs, ratings):
        self.user_item_pairs = user_item_pairs
        self.ratings = ratings

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_pairs[idx]
        rating = self.ratings[idx]
        return user_id, item_id, rating

def train_model(user_item_pairs, ratings, num_users, num_items, embedding_dim=8, epochs=10, batch_size=16):
    """
    Train the hybrid recommendation model.
    """
    dataset = SampleDataset(user_item_pairs, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HybridRecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for user_id, item_id, rating in dataloader:
            optimizer.zero_grad()
            output = model(user_id, item_id)
            loss = criterion(output, rating.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

# Simulated data
num_users = 5
num_items = 3
user_item_pairs = [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)]
ratings = [5, 3, 4, 2, 1]

train_model(user_item_pairs, ratings, num_users, num_items)