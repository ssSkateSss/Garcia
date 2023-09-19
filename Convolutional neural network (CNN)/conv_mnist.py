import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 9, 1)     # <- (9*9 + 1)*16 = 1 312
        self.conv2 = nn.Conv2d(16, 4, 9, 1)     # <- (9*9)*16*4 + 4 = 5 188
        self.dropout1 = nn.Dropout(0.25)        # <- 0
        self.dropout2 = nn.Dropout(0.5)         # <- 0
        self.fc1 = nn.Linear(144, 50)           # <- 144*50 + 50 = 7 250
        self.fc2 = nn.Linear(50, 10)            # <- 50*10 + 10 = 510

    def forward(self, x):           # [128,  1, 28, 28]
        x = self.conv1(x)           # [128, 16, 20, 20]
        x = F.relu(x)               # [128, 16, 20, 20]
        x = self.conv2(x)           # [128, 4, 12, 12]
        x = F.relu(x)               # [128, 4, 12, 12]
        x = F.max_pool2d(x, 2)      # [128, 4, 6, 6]
        x = self.dropout1(x)        # [128, 4, 6, 6]
        x = torch.flatten(x, 1)     # [128, 4x6x6 = 144]
        x = self.fc1(x)             # [128, 50]
        x = F.relu(x)               # [128, 50]
        x = self.dropout2(x)        # [128, 50]
        output = self.fc2(x)        # [128, 10]
        return output


def train(model, device, train_loader, optimizer, epoch, loss_fn):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_fn):
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

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_kwargs = {'batch_size': 128}
    test_kwargs = {'batch_size': 128}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./mnsit-dataset', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('./mnsit-dataset', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    epochs = 5

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        test(model, device, test_loader, loss_fn)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)
    print(summary(model, input_size=(1, 28, 28)))


if __name__ == '__main__':
    main()
