import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_multilabel_classification

class HybridRecommender(nn.Module):
    """
    A hybrid recommendation model combining collaborative filtering and content-based filtering.
    """
    def __init__(self, num_users, num_items, embedding_dim):
        super(HybridRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        combined = torch.cat([user_vec, item_vec], dim=1)
        return torch.sigmoid(self.fc(combined))

class RecommendationDataset(Dataset):
    """
    Custom dataset for loading user-item interactions.
    """
    def __init__(self, user_item_pairs, labels):
        self.user_item_pairs = user_item_pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_item_pairs[idx], self.labels[idx]

def train_model(model, train_loader, num_epochs=5, learning_rate=0.01):
    """
    Trains the hybrid recommendation model.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for (user_item, label) in train_loader:
            user_ids, item_ids = user_item
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), label.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

def main():
    """
    Main function to simulate data and run training.
    """
    num_users = 1000
    num_items = 500
    user_item_pairs, labels = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=1, n_labels=1, random_state=42)
    dataset = RecommendationDataset(user_item_pairs, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = HybridRecommender(num_users, num_items, embedding_dim=16)
    train_model(model, train_loader)

if __name__ == '__main__':
    main()