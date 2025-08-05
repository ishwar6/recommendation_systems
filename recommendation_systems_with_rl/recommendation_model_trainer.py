import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class RecommendationDataset(Dataset):
    """Custom dataset for recommendation system using user-item interactions."""
    def __init__(self, interactions):
        self.interactions = interactions
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_input = self.interactions[idx]['user']
        item_input = self.interactions[idx]['item']
        user_tokens = self.tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
        item_tokens = self.tokenizer(item_input, return_tensors='pt', padding=True, truncation=True)
        return user_tokens, item_tokens

class RecommendationModel(nn.Module):
    """BERT-based recommendation model."""
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 * 2, 1)

    def forward(self, user_tokens, item_tokens):
        user_output = self.bert(**user_tokens)
        item_output = self.bert(**item_tokens)
        combined = torch.cat((user_output.pooler_output, item_output.pooler_output), dim=1)
        return self.fc(combined)

def train_recommendation_model(interactions, epochs=5, batch_size=8):
    """Train the recommendation model on user-item interactions."""
    dataset = RecommendationDataset(interactions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RecommendationModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        for user_tokens, item_tokens in dataloader:
            optimizer.zero_grad()
            output = model(user_tokens[0], item_tokens[0])
            labels = torch.tensor([interaction['label'] for interaction in interactions]).float().to(output.device)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')