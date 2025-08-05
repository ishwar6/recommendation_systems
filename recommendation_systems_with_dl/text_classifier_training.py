import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True)
        label = torch.tensor(self.labels[idx])
        return inputs, label

class TextClassifier(nn.Module):
    """Simple text classification model using BERT."""
    def __init__(self, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def train_model(texts, labels, num_classes, epochs=3, batch_size=16):
    """Train the text classification model."""
    dataset = TextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TextClassifier(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, label in dataloader:
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

texts = ['This is a positive review.', 'This is a negative review.']
labels = [1, 0]
train_model(texts, labels, num_classes=2)