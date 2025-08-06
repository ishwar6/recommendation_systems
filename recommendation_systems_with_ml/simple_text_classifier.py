import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

def train_model(texts, labels, num_epochs=10, batch_size=32):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2)
    dataset_train = TextDataset(X_train, y_train)
    dataset_val = TextDataset(X_val, y_val)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    model = SimpleNN(input_size=X.shape[1], num_classes=len(set(labels)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for texts_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(texts_batch))
            loss = criterion(outputs, torch.LongTensor(labels_batch))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts_batch, labels_batch in val_loader:
            outputs = model(torch.FloatTensor(texts_batch))
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted.numpy() == labels_batch).sum()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Mock data for simulation
if __name__ == '__main__':
    mock_texts = ['I love programming', 'Python is great', 'I hate bugs', 'Debugging is fun']
    mock_labels = [1, 1, 0, 1]
    train_model(mock_texts, mock_labels)