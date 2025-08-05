import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def load_data():
    data = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball', 'sci.med'])
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    return X_train_vec, X_test_vec, vectorizer

def train_model(X_train, y_train, input_size, num_classes):
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted.numpy() == y_test).mean()
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    X_train_vec, X_test_vec, vectorizer = preprocess_data(X_train, X_test)
    model = train_model(X_train_vec, y_train, input_size=X_train_vec.shape[1], num_classes=2)
    evaluate_model(model, X_test_vec, y_test)