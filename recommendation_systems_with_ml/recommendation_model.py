import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np

class RecommendationModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.fc(pooled_output)

    def predict(self, text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            logits = self(inputs['input_ids'], inputs['attention_mask'])
        return torch.sigmoid(logits).numpy()

# Sample usage and output demonstration
if __name__ == '__main__':
    model = RecommendationModel()
    sample_text = ["This course is fantastic!", "I didn't find it helpful."]
    predictions = model.predict(sample_text)
    print(predictions)