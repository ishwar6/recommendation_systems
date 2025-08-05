import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

class TextDataset(Dataset):
    """Custom Dataset for loading text data for classification."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True, max_length=512)
        return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'labels': torch.tensor(self.labels[idx])}

class TextClassifier(nn.Module):
    """BERT-based text classifier."""
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)

def train_model(texts, labels, epochs=3):
    """Train the text classification model."""
    dataset = TextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TextClassifier().train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    print('Training complete.')

# Example usage with mock data
if __name__ == '__main__':
    mock_texts = ['This is a positive example.', 'This is a negative example.']
    mock_labels = [1, 0]
    train_model(mock_texts, mock_labels)