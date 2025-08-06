import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), label

class RecommendationModel(nn.Module):
    """Recommendation model based on BERT architecture."""
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state[:, 0, :])
        return logits

def train_model(texts, labels, epochs=5, batch_size=16):
    """Function to train the recommendation model."""
    dataset = TextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RecommendationModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(epochs):
        for input_ids, attention_mask, label in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

texts = ['I love this course!', 'This was a waste of time.', 'Highly recommend this class.']
labels = [1, 0, 1]
train_model(texts, labels)