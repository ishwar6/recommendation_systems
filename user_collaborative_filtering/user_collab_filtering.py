import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class UserCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(UserCollaborativeFiltering, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        return (user_embeds * item_embeds).sum(1)

def train_model(data, num_users, num_items, embedding_dim=8, epochs=10, lr=0.01):
    user_ids, item_ids, ratings = data
    user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(
        user_ids, item_ids, ratings, test_size=0.2, random_state=42)
    model = UserCollaborativeFiltering(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(user_ids_train, item_ids_train)
        loss = criterion(predictions, ratings_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return model

if __name__ == '__main__':
    num_users = 100
    num_items = 50
    user_ids = np.random.randint(0, num_users, size=1000)
    item_ids = np.random.randint(0, num_items, size=1000)
    ratings = np.random.rand(1000) * 5
    data = (torch.tensor(user_ids), torch.tensor(item_ids), torch.tensor(ratings, dtype=torch.float32))
    model = train_model(data, num_users, num_items)