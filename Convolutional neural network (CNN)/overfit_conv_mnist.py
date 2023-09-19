import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 528)          # <- 784*528 + 528 = 414 480
        self.fc2 = nn.Linear(528, 128)          # <- 528*128 + 128 = 67 712
        self.fc3 = nn.Linear(128, 10)           # <- 128*10 + 10 = 1 290

    def forward(self, x):           
        x = torch.flatten(x, 1)     
        x = self.fc1(x)             
        x = F.relu(x)               
        x = self.fc2(x)             
        x = F.relu(x)               
        output = self.fc3(x)        
        return output


def main():

    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size': 2}
    test_kwargs = {'batch_size': 2}

    from torch.utils.data import DataLoader, SubsetRandomSampler
   

    # Створення трансформацій та параметрів завантаження
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Завантаження датасету MNIST
    train_dataset = datasets.MNIST('./mnsit-dataset', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('./mnsit-dataset', train=False,
                              transform=transform)

    # Міні датасет MNIST
    samples_per_class = 100
    class_indices = [[] for _ in range(10)]
    for idx, (data, target) in enumerate(train_dataset):
        if len(class_indices[target]) < samples_per_class:
            class_indices[target].append(idx)
    selected_indices = [idx for indices in class_indices for idx in indices]
    sampler = SubsetRandomSampler(selected_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, **train_kwargs)

    samples2_per_class = 20
    class_indices = [[] for _ in range(10)]
    for idx, (data, target) in enumerate(test_dataset):
        if len(class_indices[target]) < samples2_per_class:
            class_indices[target].append(idx)
    selected_indices = [idx for indices in class_indices for idx in indices]
    sampler = SubsetRandomSampler(selected_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, **test_kwargs)   

    epochs = 5

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses = [] 
    test_losses = []
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 2*len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # sum up batch loss
                test_loss += loss_fn(output, target).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= 2*len(test_loader)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, 2*len(test_loader),
            100. * correct / 200))
        scheduler.step()
        

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)
    print(summary(model, input_size=(1, 28, 28)))

    plt.subplots(figsize=(8, 6))
    plt.title('Loss')
    plt.plot(range(1, epochs+1), np.log(train_losses), label='train')
    plt.plot(range(1, epochs+1), np.log(test_losses), label='test')
    plt.legend()
    plt.savefig('Train and test loss.png')

    

if __name__ == '__main__':
    main()

