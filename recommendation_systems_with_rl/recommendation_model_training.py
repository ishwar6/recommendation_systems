import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np

class RecommendationModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        return self.fc(pooled_output)

    def train_model(self, dataset, epochs=3, batch_size=8):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(dataset['text'].tolist(), return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(dataset['label'].tolist()).unsqueeze(1).float()
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], labels, test_size=0.2)
        train_attn_masks, val_attn_masks = train_test_split(inputs['attention_mask'], test_size=0.2)
        model = self
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(train_inputs), batch_size):
                optimizer.zero_grad()
                inputs_batch = train_inputs[i:i+batch_size]
                masks_batch = train_attn_masks[i:i+batch_size]
                labels_batch = train_labels[i:i+batch_size]
                outputs = model(inputs_batch, masks_batch)
                loss = nn.MSELoss()(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        self.evaluate(val_inputs, val_attn_masks, val_labels)

    def evaluate(self, val_inputs, val_attn_masks, val_labels):
        self.eval()
        with torch.no_grad():
            outputs = self(val_inputs, val_attn_masks)
            preds = outputs.detach().numpy()
            mse = np.mean((preds - val_labels.numpy()) ** 2)
            print(f'Validation MSE: {mse}')

# Sample data simulation and usage:
data = {'text': ['This is a great course', 'I loved the class', 'Not worth the time', 'Amazing experience'], 'label': [5, 4, 2, 5]}
dataset = pd.DataFrame(data)
model = RecommendationModel('bert-base-uncased')
model.train_model(dataset)