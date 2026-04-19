import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

seed = 42
torch.manual_seed(seed)
#random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))

print("datatype of the 1st training sample: ", train_dataset[0][0].type())
print("size of the 1st training sample: ", train_dataset[0][0].size())

batch_size = 64

# Create data loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Hardware
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define the model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

models = []
for i in range(5):
    models.append(LeNet5().to(device))


count_parameters(models[0])
fc12_params = [p for name, p in models[0].named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizers = []
optimizers.append(optim.AdamW(models[0].parameters(), lr=0.001))
optimizers.append(optim.AdamW(models[1].parameters(), lr=0.001))
optimizers.append(optim.AdamW(models[2].parameters(), lr=0.001))
optimizers.append(optim.AdamW(models[3].parameters(), lr=0.001))
optimizers.append(optim.AdamW(models[4].parameters(), lr=0.001))


modelSTD = LeNet5().to(device)
optimizerSTD = optim.SGD(modelSTD.parameters(), lr=0.001)


n_epochs = 30
train_losses = torch.empty(len(optimizers)+1, n_epochs, 2)
test_losses = torch.empty(len(optimizers)+1, n_epochs, 2)
accuracy = torch.empty(len(optimizers)+1, n_epochs, 2)
print(train_losses.shape)
print(test_losses.shape)

batches = [16, 32, 64, 128, 256]
warmup_epochs_factor = [10, 20, 30, 40, 50] 
schedulers = []

"""
for i, optimizer in enumerate(optimizers):
    warmup_epochs = n_epochs / warmup_epochs_factor[i]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs # low lr as warmup
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max= n_epochs - warmup_epochs                               # do CosineAnnealingLR
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    schedulers.append(scheduler)
    """



for j, model in enumerate(models):
    batch_size =  batches[j]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    test_loss_epoch = []
    train_loss_epoch = []
    accuracy_epoch = []
    step = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizers[j].zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            train_loss += criterion(output, labels).item()
            loss.backward()
            optimizers[j].step()
            step += 1
            if (i % 500 == 0):
               print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

        train_loss /= len(train_loader.dataset)
        train_loss_epoch.append((epoch+1, loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                test_loss += criterion(output, labels).item()
                pred = torch.argmax(output, dim=1)
                correct += pred.eq(labels).sum()

        # schedulers[j].step()
        test_loss /= len(test_loader.dataset)
        test_loss_epoch.append((epoch+1, test_loss))
        accuracy_epoch.append((epoch+1, 100. * correct / len(test_loader.dataset)))
        print(f'Test set: Average loss: {test_loss}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
        
    print(j)
    #print(f"Test_losses size: {test_losses[j,k].shape}")
    # print(f"Test_losses input size: {torch.tensor(test_loss_epoch, dtype=torch.float32).shape}")
    # print(f"Train_losses size: {train_losses[j,k].shape}")
    # print(f"Train_losses input size: {torch.tensor(train_loss_epoch, dtype=torch.float32).shape}")
    test_losses[j] = torch.tensor(test_loss_epoch, dtype=torch.float32)
    train_losses[j] = torch.tensor(train_loss_epoch, dtype=torch.float32)
    accuracy[j] = torch.tensor(accuracy_epoch, dtype=torch.float32)

"""
### TEST CONTROL
test_loss_epoch = []
train_loss_epoch = []
accuracy_epoch = []
step = 0
for epoch in range(n_epochs):
    modelSTD.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizerSTD.zero_grad()
        output = modelSTD(images)
        loss = criterion(output, labels)
        train_loss += criterion(output, labels).item()
        loss.backward()
        optimizerSTD.step()
        step += 1
        if (i % 500 == 0):
           print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    train_loss /= len(train_loader.dataset)
    train_loss_epoch.append((epoch+1, loss.item()))

    modelSTD.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = modelSTD(images)
            test_loss += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    # schedulers[j].step()
    test_loss /= len(test_loader.dataset)
    test_loss_epoch.append((epoch+1, test_loss))
    accuracy_epoch.append((epoch+1, 100. * correct / len(test_loader.dataset)))
    print(f'Test set: Average loss: {test_loss}, \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
    
#print(f"Test_losses size: {test_losses[j,k].shape}")
# print(f"Test_losses input size: {torch.tensor(test_loss_epoch, dtype=torch.float32).shape}")
# print(f"Train_losses size: {train_losses[j,k].shape}")
# print(f"Train_losses input size: {torch.tensor(train_loss_epoch, dtype=torch.float32).shape}")
test_losses[5] = torch.tensor(test_loss_epoch, dtype=torch.float32)
train_losses[5] = torch.tensor(train_loss_epoch, dtype=torch.float32)
accuracy[5] = torch.tensor(accuracy_epoch, dtype=torch.float32)
"""






"""
# plot train and test losses to file loss.png
train_steps_SGD, train_loss_SGD = zip(*train_losses_SGD)
test_steps_SGD, test_loss_SGD = zip(*test_losses_SGD)
train_steps_AdamW, train_loss_AdamW = zip(*train_losses_AdamW)
test_steps_AdamW, test_loss_AdamW = zip(*test_losses_AdamW)
"""

fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)  # sharex aligns x-axes

# Plot Train Loss
ax[0].plot(train_losses[0, :, 0], train_losses[0, :, 1], label="Train Loss - Batch Size 16", color="blue")
ax[0].plot(train_losses[1, :, 0], train_losses[1, :, 1], label="Train Loss - Batch Size 32", color="red")
ax[0].plot(train_losses[2, :, 0], train_losses[2, :, 1], label="Train Loss - Batch Size 64", color="green")
ax[0].plot(train_losses[3, :, 0], train_losses[3, :, 1], label="Train Loss - Batch Size 128", color="yellow")
ax[0].plot(train_losses[4, :, 0], train_losses[4, :, 1], label="Train Loss - Batch Size 256", color="cyan")
ax[0].set_ylabel("Loss")
ax[0].legend(loc="upper right")
ax[0].grid(True)

# Plot Test Accuracy
ax[1].plot(accuracy[0, :, 0], accuracy[0, :, 1], label="Accuracy - Batch Size 16", color="blue")
ax[1].plot(accuracy[1, :, 0], accuracy[1, :, 1], label="Accuracy - Batch Size 32", color="red")
ax[1].plot(accuracy[2, :, 0], accuracy[2, :, 1], label="Accuracy - Batch Size 64", color="green")
ax[1].plot(accuracy[3, :, 0], accuracy[3, :, 1], label="Accuracy - Batch Size 128", color="yellow")
ax[1].plot(accuracy[4, :, 0], accuracy[4, :, 1], label="Accuracy - Batch Size 256", color="cyan")
ax[1].set_ylabel("Test Accuracy")
ax[1].legend(loc="lower right")
ax[1].grid(True)

# Plot Test Loss
ax[2].plot(test_losses[0, :, 0], test_losses[0, :, 1], label="Test Loss - Batch Size 16", color="blue")
ax[2].plot(test_losses[1, :, 0], test_losses[1, :, 1], label="Test Loss - Batch Size 32", color="red")
ax[2].plot(test_losses[2, :, 0], test_losses[2, :, 1], label="Test Loss - Batch Size 64", color="green")
ax[2].plot(test_losses[3, :, 0], test_losses[3, :, 1], label="Test Loss - Batch Size 128", color="yellow")
ax[2].plot(test_losses[4, :, 0], test_losses[4, :, 1], label="Test Loss - Batch Size 256", color="cyan")

ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Loss")
ax[2].legend()
ax[2].grid(True)
ax[2].legend(loc="upper right")
# Adjust layout and show the plot
plt.tight_layout()
# plt.savefig('Comparing_batchSizes_AdamW.png')
plt.show()

