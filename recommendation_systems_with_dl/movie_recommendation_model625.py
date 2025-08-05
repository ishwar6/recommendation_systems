import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class MovieRecommendationModel(nn.Module):
    """Model for recommending movies based on user input."""
    def __init__(self):
        super(MovieRecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass for the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

class MovieDataset(Dataset):
    """Dataset for loading movie titles for recommendations."""
    def __init__(self, titles):
        self.titles = titles
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        inputs = self.tokenizer(title, return_tensors='pt', padding='max_length', max_length=32, truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

def train_model(titles):
    """Train the movie recommendation model on provided titles."""
    dataset = MovieDataset(titles)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = MovieRecommendationModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        for input_ids, attention_mask in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            labels = torch.ones(outputs.size(), device=outputs.device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    return model

if __name__ == '__main__':
    sample_titles = ['Inception', 'The Matrix', 'Interstellar', 'The Godfather']
    trained_model = train_model(sample_titles)  
    print("Training completed and model is ready for inference.")