import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np

class TextDataset(Dataset):
    """Custom Dataset for loading text data for recommendation system."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), self.labels[idx]

class RecommendationModel(nn.Module):
    """BERT-based model for generating item recommendations."""
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits.squeeze()

def train_model(texts, labels, epochs=5, batch_size=8):
    """Train the recommendation model using provided texts and labels."""
    model = RecommendationModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)
    train_dataset = TextDataset(texts_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for input_ids, attention_mask, label in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

texts = ['I love reading books.', 'Machine learning is fascinating.', 'Deep learning enables amazing applications.']
labels = [1, 1, 0]
train_model(texts, labels)