import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class RecommendationModel(nn.Module):
    """A recommendation model using BERT embeddings."""
    def __init__(self, dropout_rate=0.1):
        super(RecommendationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropped_out = self.dropout(pooled_output)
        return self.fc(dropped_out)

def prepare_data(texts):
    """Tokenize input texts and create input tensors."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']

if __name__ == '__main__':
    sample_texts = ['This is a great course!', 'I did not enjoy this class.']
    input_ids, attention_mask = prepare_data(sample_texts)
    model = RecommendationModel()
    output = model(input_ids, attention_mask)
    print(output.detach().numpy())