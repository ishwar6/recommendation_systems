import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class MovieRecommendationDataset(Dataset):
    """
    Custom Dataset for loading movie reviews and ratings.
    """
    def __init__(self, reviews, ratings, tokenizer, max_length):
        self.reviews = reviews
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        rating = self.ratings[index]
        inputs = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        return inputs['input_ids'].flatten(), inputs['attention_mask'].flatten(), rating

class RecommendationModel(nn.Module):
    """
    A simple recommendation model using BERT to encode reviews.
    """
    def __init__(self, dropout_rate=0.3):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        return self.fc(dropped_output)

def train_model(reviews, ratings, epochs=3, batch_size=16):
    """
    Train the recommendation model on movie reviews.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = MovieRecommendationDataset(reviews, ratings, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RecommendationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, rating in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.flatten(), rating.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

# Simulated data
if __name__ == '__main__':
    mock_reviews = ['Great movie!', 'Not my favorite.', 'Loved it!', 'It was okay.']
    mock_ratings = [5, 2, 5, 3]
    train_model(mock_reviews, mock_ratings)