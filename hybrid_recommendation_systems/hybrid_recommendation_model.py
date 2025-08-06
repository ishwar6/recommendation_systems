import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

class HybridRecommendationSystem:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

    def train_model(self, texts, ratings, epochs=3):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=128,
                                                 padding='max_length', return_attention_mask=True,
                                                 return_tensors='pt', truncation=True)
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        input_ids = torch.cat(input_ids)
        attention_masks = torch.cat(attention_masks)
        ratings = torch.tensor(ratings, dtype=torch.float32).view(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(input_ids, ratings, test_size=0.2)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.forward(X_train, attention_masks[:X_train.size(0)])
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def recommend(self, text):
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=128,
                                                 padding='max_length', return_attention_mask=True,
                                                 return_tensors='pt', truncation=True)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            output = self.forward(input_ids, attention_mask)
            return output.item()

# Example Usage
if __name__ == '__main__':
    texts = ['This is great!', 'I did not like this.', 'Amazing experience.']
    ratings = [5, 1, 4]
    recommender = HybridRecommendationSystem()
    recommender.train_model(texts, ratings)
    print(recommender.recommend('Fantastic service!'))