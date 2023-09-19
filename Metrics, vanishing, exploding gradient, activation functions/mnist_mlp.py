"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""


from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):
    
    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int, dropout_prob: float = 0.5) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.wih = torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=hiddennodes)
        self.whh = torch.nn.Linear(in_features=hiddennodes, out_features=hiddennodes)
        self.who = torch.nn.Linear(in_features=hiddennodes, out_features=outputnodes)
        self.activation = torch.nn.Hardswish()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wih(x)
        out = self.activation(out)
        out = self.BatchNorm(out)
        out = self.dropout(out)
        out = self.whh(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.who(out)
        return out


class MnistDataset(Dataset):
    
    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        # conver string data to torch Tensor data type
        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)

    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


if __name__ == "__main__":
    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    batch_size = 10
    # number of epochs
    epochs = 5

    train_losses = []
    test_losses = []

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="./mnist_train.csv")
    test_dataset = MnistDataset(filepath="./mnist_test.csv")

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Define NN
    model = MnistMlp(
                     inputnodes=input_nodes, 
                     hiddennodes=hidden_nodes, 
                     outputnodes=output_nodes
                     )
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)
    
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ##### Training! #####
    model.train()
    for epoch in range(epochs):
        for batch_idx, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        train_losses.append(loss.item())
      
    

        ##### Testing! #####
        model.eval()
        test_loss = 0
        correct = 0
        true_labels = []
        predicted_labels = []
        with torch.inference_mode():
            for features, target in test_loader:
                features, target = features.to(device), target.to(device)
                output = model(features)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                predicted_labels.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
       
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        


    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")


    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate and print accuracy, precision, recall, and F1-score
    print(f"Accuracy Score:\n general: {accuracy_score(true_labels, predicted_labels)}\n")
    print(f"Precision Score:\n per class: {precision_score(true_labels, predicted_labels, average=None)}\n general: {precision_score(true_labels, predicted_labels, average='weighted')}\n")
    print(f"Recall Score:\n per class: {recall_score(true_labels, predicted_labels, average=None)}\n general: {recall_score(true_labels, predicted_labels, average='weighted')}\n")
    print(f"F1-Score:\n per class: {f1_score(true_labels, predicted_labels, average=None)}\n general: {f1_score(true_labels, predicted_labels, average='weighted')}\n") 

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                            display_labels=class_labels)

    disp.plot(cmap=matplotlib.colormaps["hot"])
    plt.show()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis],
                              display_labels=class_labels)
    disp.plot(cmap=matplotlib.colormaps["hot"])
    plt.show()

    # Generate and print classification report
    class_report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report:")
    print(class_report)
    

plt.subplots(figsize=(8, 6))
plt.title('Loss')
plt.plot(range(1, epochs+1), np.log(train_losses), label='train')
plt.plot(range(1, epochs+1), np.log(test_losses), label='test')
plt.legend()
plt.savefig('Train and test loss.png')

