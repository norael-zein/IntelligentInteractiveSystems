import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch

#Neural network class
class Neural_Network(nn.Module):
    def __init__(self, features_in=21, features_out=7): #21 different AU and 7 different emotions
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, features_out)
        )
    
    def forward(self, input):
        return self.net(input)

#Manage dataset
class EmotionDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        #Read input file
        data = pd.read_csv(data_path)
        
        #Drop emotions from data
        self.inputs = torch.tensor(data.drop("emotion", axis=1).to_numpy(dtype=np.float32))
        
        #emotions become numerical values
        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}
        self.labels = torch.tensor(data["emotion"].apply(lambda x: label2index[x]).to_numpy())

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)

# Run training, validation and test
def main():
    dataset = EmotionDataset("dataset/test/dataset.csv")
    
    # Split data: 70% for training, 20% for validation, 10% for testing
    generator = torch.Generator().manual_seed(2023)
    train, val, test = random_split(dataset, [0.7, 0.2, 0.1], generator=generator)
    
    # Data loaders
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=4) 
    test_loader = DataLoader(test, batch_size=4)
    
    # Create our model
    model = Neural_Network(train[0][0].shape[0], len(dataset.index2label))
    optim = torch.optim.Adam(model.parameters(), lr=0.005)  # Adam optimizer, learning rate 0.005
    loss_fn = nn.CrossEntropyLoss()
    
    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Training loop
    for epoch in range(30):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()  
            outputs = model(inputs)  
            loss = loss_fn(outputs, labels)  
            loss.backward() 
            optim.step()  
        
        # Validation after each epoch
        model.eval()
        print(f"Epoch {epoch+1}")
        with torch.no_grad():
            correct_val = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                correct_val += (predictions.argmax(dim=1) == labels).sum()
            val_accuracy = correct_val / len(val)
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
            
        
        # Test after each epoch (or at the end of training)
        with torch.no_grad():
            correct_test = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                correct_test += (predictions.argmax(dim=1) == labels).sum()
            test_accuracy = correct_test / len(test)
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

