import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SimpleData(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.X = torch.randn(num_samples, 10)
        self.y = torch.randint(0, 2, (num_samples,))

    def __len__ (self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SimpleModel(nn.Module):
    def __init__ (self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
input_size = 10
hidden_size = 7
output_size = 1
learning_rate = 0.001
num_epochs = 10
batch_size = 32

dataset = SimpleData(num_samples=200)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)

        labels = labels.float().unsqueeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


model.eval()

with torch.no_grad():
    test_inputs, test_labels = dataset[0:10]
    test_labels.float().unsqueeze(1)

    predictions = model(test_inputs)

    predicted_labels = (predictions > 0.5).float()

    accuracy = (predicted_labels == test_labels).float().mean()

    print(f"Accuracy on first 10 samples: {accuracy:.4f}")

