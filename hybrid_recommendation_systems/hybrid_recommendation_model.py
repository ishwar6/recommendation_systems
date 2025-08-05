import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class HybridRecommendationModel(nn.Module):
    """
    A hybrid recommendation model combining collaborative filtering and content-based filtering.
    """
    def __init__(self, num_users, num_items, embedding_dim):
        super(HybridRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        concatenated = torch.cat([user_vec, item_vec], dim=1)
        return self.fc(concatenated)

class MockDataset(Dataset):
    """
    A mock dataset for user-item interactions.
    """
    def __init__(self, num_users, num_items, num_samples):
        self.user_ids = np.random.randint(0, num_users, num_samples)
        self.item_ids = np.random.randint(0, num_items, num_samples)
        self.ratings = np.random.rand(num_samples)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.item_ids[idx], self.ratings[idx])

def train_model(num_users, num_items, embedding_dim, num_samples, epochs=10, batch_size=32):
    """
    Trains the hybrid recommendation model on mock data.
    """
    dataset = MockDataset(num_users, num_items, num_samples)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HybridRecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for user_ids, item_ids, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids).squeeze()
            loss = criterion(outputs, ratings.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

if __name__ == '__main__':
    trained_model = train_model(num_users=1000, num_items=500, embedding_dim=20, num_samples=10000, epochs=5, batch_size=64)