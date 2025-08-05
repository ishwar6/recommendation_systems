import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleRecommendationModel(nn.Module):
    """
    A simple neural network model for item recommendation based on user-item interactions.
    """
    def __init__(self, num_users, num_items, embedding_dim):
        super(SimpleRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return (user_embeds * item_embeds).sum(1)

class InteractionDataset(Dataset):
    """
    Dataset class for loading user-item interactions.
    """
    def __init__(self, user_item_pairs):
        self.user_item_pairs = user_item_pairs

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_pairs[idx]
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(item_id, dtype=torch.long)

def train_model(user_item_pairs, num_users, num_items, embedding_dim=8, epochs=5, batch_size=32):
    """
    Trains the recommendation model on provided user-item interactions.
    """
    model = SimpleRecommendationModel(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_data, val_data = train_test_split(user_item_pairs, test_size=0.2)
    train_loader = DataLoader(InteractionDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(InteractionDataset(val_data), batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for user_ids, item_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, torch.ones(user_ids.size(0)))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return model

if __name__ == '__main__':
    mock_data = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1)]
    trained_model = train_model(mock_data, num_users=3, num_items=3)