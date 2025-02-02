import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch

"""
This is a machine learning pipeline of a Neural Network that predicts emotions based on Arousal, Valence 
and Action Units. 

INPUT: facial_features_original.csv (Arousal, Valence and Action Units)
OUTPUT: Neural network with lowest loss, saved in models/emotion_model_nn.pth
"""

#Neural network class
class Neural_Network(nn.Module):
    def __init__(self, features_in=22, features_out=7):  #22 different AU+valence and arousal and 7 different emotions
        super().__init__()
        #Sequential neural network with linear layers
        self.net = nn.Sequential(
            nn.Linear(features_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),  #Regularization
            nn.Linear(16, features_out)
        )
    
    def forward(self, input):
        return self.net(input)

#Manage our dataset
class EmotionDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        
        #Read input file
        data = pd.read_csv(data_path)
        
        #Drop emotions from data
        self.inputs = torch.tensor(data.drop(["subDirectory_filePath", "expression"], axis=1).to_numpy(dtype=np.float32))
        
        #Emotions become numerical values
        self.index2label = [label for label in data["expression"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}
        self.labels = torch.tensor(data["expression"].apply(lambda x: label2index[x]).to_numpy())

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)

#Run training, validation and test
def main():
    dataset = EmotionDataset("processed/facial_features_original.csv")
    
    #Split data: 70% for training, 20% for validation, 10% for testing
    generator = torch.Generator().manual_seed(2023)
    train, val, test = random_split(dataset, [0.7, 0.2, 0.1], generator=generator)
    
    #Data loaders for train, validation and test
    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=4) 
    test_loader = DataLoader(test, batch_size=4)
    
    #Create our model
    model = Neural_Network(train[0][0].shape[0], len(dataset.index2label))
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer, learning rate 0.0001
    loss_fn = nn.CrossEntropyLoss()
    
    #Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    #Regularization with early stopping 
    best_val_loss = float("inf")
    patience = 5  #Number of epochs to wait for improvement
    patience_counter = 0
    best_model_wts = model.state_dict()  #Save the best model
    best_epoch = -1  #Track the epoch where the best model occurred
    
    #Training loop
    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            loss = loss_fn(outputs, labels)  
            
            loss.backward() 
            optim.step() 
            optim.zero_grad() 
        
        #Validation after each epoch
        model.eval()
        print(f"Epoch {epoch+1}")

        with torch.no_grad():
            correct_val = 0
            val_loss = 0  #Track validation loss
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                val_loss += loss_fn(predictions, labels).item() 
                correct_val += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()

            #Validation loss of current model
            val_loss /= len(val) 
            val_accuracy = correct_val / len(val)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
            
            #Early stopping check. Check if current validation loss is smaller than the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_model_wts = model.state_dict()  #Save the best model
                best_epoch = epoch + 1  #Save the best epoch
                patience_counter = 0  #Reset counter
            else:
                patience_counter += 1 #If our loss is greater than the best validation loss, continue
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    model.load_state_dict(best_model_wts)  #Load the best model
                    break


    #Load the best model for final evaluation
    model.load_state_dict(best_model_wts)
    
    #Evaluate the test accuracy on the best model
    model.eval()
    correct_test = 0
    with torch.no_grad():
        correct_test = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            correct_test += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()
        test_accuracy = correct_test / len(test)
        print(test_accuracy)

    
    #Save the best model at the end of training
    model.load_state_dict(best_model_wts) 
    torch.save(model.state_dict(), "scripts/subSystem1/models/emotion_model_nn.pth")

    #General model information
    model_info = {
        "model": model,
        "best_weights": best_model_wts,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }    
    print(model_info)


if __name__ == "__main__":
    main()
