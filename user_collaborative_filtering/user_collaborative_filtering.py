import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

class UserCollaborativeFiltering:
    def __init__(self, num_users, num_items, embedding_dim=8):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        return (user_vecs * item_vecs).sum(1)

    def train_model(self, user_ids, item_ids, ratings, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.forward(user_ids, item_ids)
            loss = self.loss_function(predictions, ratings)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def simulate_data(num_users, num_items, num_samples):
    user_ids = np.random.randint(0, num_users, num_samples)
    item_ids = np.random.randint(0, num_items, num_samples)
    ratings = np.random.rand(num_samples) * 5
    return torch.LongTensor(user_ids), torch.LongTensor(item_ids), torch.FloatTensor(ratings)

if __name__ == '__main__':
    num_users = 100
    num_items = 50
    user_ids, item_ids, ratings = simulate_data(num_users, num_items, 1000)
    model = UserCollaborativeFiltering(num_users, num_items)
    model.train_model(user_ids, item_ids, ratings)