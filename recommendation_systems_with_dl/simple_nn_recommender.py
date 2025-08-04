import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SimpleRecommendationDataset(Dataset):
    def __init__(self, user_item_pairs):
        self.user_item_pairs = user_item_pairs

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(SimpleNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        dot_product = (user_emb * item_emb).sum(1)
        return self.fc(dot_product)

def train_recommendation_model(user_item_pairs, num_users, num_items, embedding_dim, epochs=5, batch_size=32):
    dataset = SimpleRecommendationDataset(user_item_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleNN(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for user, item in dataloader:
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output.view(-1), torch.ones(output.size(0)))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')  
    return model

if __name__ == '__main__':
    user_item_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)]
    train_recommendation_model(user_item_pairs, num_users=3, num_items=3, embedding_dim=4, epochs=10, batch_size=2)