import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import joblib


#Neural network class
class Neural_Network(nn.Module):
    def __init__(self, features_in=22, features_out=7):  #22 different AU+valence and arousal and 7 different emotions
        super().__init__()
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

            val_loss /= len(val)  #DOUBLE CHECK IF THIS SHOULD BE DIVIDED BY ONE BATCH OR WHOLE DATASET
            val_accuracy = correct_val / len(val)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
            
            #Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_model_wts = model.state_dict()  #Save the best model
                best_epoch = epoch + 1  #Save the best epoch
                patience_counter = 0  #Reset counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    model.load_state_dict(best_model_wts)  #Load the best model
                    break
  


    #Test after last epoch and for the epoch corresponding to the best epoch  - how do I take the test accuracy on the best model?
    with torch.no_grad():
        correct_test = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = best_model_wts(inputs)
            correct_test += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()
        test_accuracy = correct_test / len(test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
    #Save the best model at the end of training
    model.load_state_dict(best_model_wts) 
    joblib.dump(model.state_dict(), "scripts/subSystem1/best_model_neural_network.pkl")
    print(f"Best model had a validation loss of: {best_val_loss:.2f} and est validation accuracy is {best_val_accuracy * 100:.2f}%")
    print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
