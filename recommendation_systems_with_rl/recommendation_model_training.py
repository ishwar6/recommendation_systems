import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class RecommendationDataset(Dataset):
    def __init__(self, texts, ratings):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]
        tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), rating

class RecommendationModel(nn.Module):
    def __init__(self):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits.squeeze()

def train_model(texts, ratings, epochs=3, batch_size=4):
    dataset = RecommendationDataset(texts, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RecommendationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, rating in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, rating.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

texts = ['I love this course!', 'This was terrible.', 'Great content and examples!', 'Not worth the time.']
ratings = [5.0, 1.0, 4.5, 2.0]
train_model(texts, ratings)