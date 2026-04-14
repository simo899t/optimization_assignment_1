import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from bs_scheduler import StepBS
import time


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

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    #random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True  # windows test

    # Download the MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    print("number of training samples: " + str(len(train_dataset)) + "\n" +
          "number of testing samples: " + str(len(test_dataset)))

    print("datatype of the 1st training sample: ", train_dataset[0][0].type())
    print("size of the 1st training sample: ", train_dataset[0][0].size())

    batch_size = 512
    max_batch_size = 60000
    print(f"starting batch_size: ", batch_size, " and growing to: ", max_batch_size)

    # Hardware
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Create data loaders.
    num_w = 4 if device == 'cuda' else 0
    pin = device == 'cuda'
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=pin, persistent_workers=(num_w > 0))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=pin, persistent_workers=(num_w > 0))

    # Verify size of batches
    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    start = time.perf_counter()
    model = LeNet5().to(device)
    count_parameters(model)
    fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
    print(fc12_params[0].numel())
    print(fc12_params[1].numel())

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0255, weight_decay=3e-2)
    n_epochs = 500
    l2_lambda = 1e-2
    train_losses = []
    test_losses = []
    batch_scheduler = StepBS(train_loader, step_size=20, gamma=1.5, max_batch_size=max_batch_size)


    warmup_epochs = 2
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs  # low lr as warmup
        )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs- warmup_epochs                               # do CosineAnnealingLR
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )


    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None # windows test

    step = 0
    for epoch in range(n_epochs):
        model.train()
        avg_train_loss = 0.0
        for j, (images, labels) in enumerate(train_loader):
            #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True) # windows test
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):  # windows test
                output = model(images)                                                # windows test
                loss = criterion(output, labels)
                for layer in [model.fc1, model.fc2, model.fc3]:
                    loss += l2_lambda * torch.norm(layer.weight, p=2)
            if scaler:  # windows test
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            step += 1
            avg_train_loss += loss.item()
            # if step % 10 == 0:
            print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

        avg_train_loss /= len(train_loader)
        train_losses.append((epoch+1, avg_train_loss))

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

        scheduler.step()
        batch_scheduler.step()
        test_loss /= len(test_loader.dataset)
        test_losses.append((epoch+1, test_loss))
        print(f'Test set: Average loss: {test_loss}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')

    stop = time.perf_counter()
    print(f"Time taken: {stop-start}")

    # plot train and test losses to file loss.png
    train_epochs, train_loss = zip(*train_losses)   
    test_epochs, test_loss = zip(*test_losses)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # sharex aligns x-axes

    # interval = int(np.ceil(len(train_dataset)/batch_size))
    # tick_positions = list(range(0, interval*(n_epochs+1), interval))
    # tick_labels = list(range(n_epochs+1))

    # Plot the first
    ax[0].plot(train_epochs, train_loss, label="Train Loss", color="blue")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Plot the second
    ax[1].plot(test_epochs, test_loss, label="Test Loss, Average", color="red")
    # interval = int(np.ceil(len(train_dataset)/batch_size))
    # ax[1].set_xticks(range(0,interval*(n_epochs+1),interval))
    # ax[1].set_xticklabels(range(n_epochs+1))
    tick_step = max(1, n_epochs // 10)
    ax[1].set_xticks(range(tick_step, n_epochs + 1, tick_step))
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('AdamW_lr0.01_10epochs_with_annealing_and_weight_decay.png')

### Tests ###
# loss    | acc    | Epocs | tim | lr     | Loss/Decay | batchS | sS | gam | mxBtchSiz | staBa
#----------------------------------------------------------------------------------------------
# 0.00445 | 89.809 | 20    | 51  | 0.027  | D2e-3      | stepBS | 1  | 2   | 60.000    | 64
# 0.00888 | 90.099 | 10    | n/a | 0.027  | D2e-3      | stepBS | 2  | 2   | 256       | 32
# 0.00893 | 90.059 | 10    | n/a | 0.027  | D2e-3      | stepBS | 2  | 2   | 128       | 32
# 0.00434 | 90.459 | 10    | n/a | 0.025  | D2e-3      | n/a    | x  | x   | x         | 64
# 0.00438 | 90.230 | 10    | n/a | 0.300  | D2e-3      | n/a    | x  | x   | x         | 64
# 0.00428 | 89.900 | 10    | n/a | 0.015  | D2e-3      | n/a    | x  | x   | x         | 64
# 0.00599 | 85.860 | 10    | n/a | 0.100  | D2e-3      | n/a    | x  | x   | x         | 64
# 0.00435 | 89.889 | 10    | n/a | 0.010  | D2e-3      | n/a    | x  | x   | x         | 64
# 0.00425 | 90.610 | 10    | 44  | 0.0255 | D2e-3      | stepBS | 5  | 2   | 256       | 64
# 0.00421 | 90.540 | 10    | 45  | 0.0255 | D5e-3      | stepBS | 5  | 2   | 256       | 64
# 0.00421 | 90.529 | 10    | 47  | 0.0255 | L2_1e-5    | stepBS | 5  | 2   | 256       | 64
# 0.00428 | 90.470 | 10    | 47  | 0.0255 | L2_2e-5    | stepBS | 5  | 2   | 256       | 64



### TIME ###
# WITH DEVICE
# optimizer = optim.AdamW(model.parameters(), lr=0.027, weight_decay=2e-3)
# batch_scheduler = StepBS(train_loader, step_size=2, gamma=2, max_batch_size=256)
# Tests:    Average loss:           Accuracy:           Time taken:
# 1         0.00888083192           90.099998%          47.75820712
# 2         0.00888083192           90.099998%          48.86876808
# 3         0.00888083192           90.099998%          46.69077666

# WITHOUT DEVICE
# optimizer = optim.AdamW(model.parameters(), lr=0.027, weight_decay=2e-3)
# batch_scheduler = StepBS(train_loader, step_size=2, gamma=2, max_batch_size=256)
# Tests:    Average loss:           Accuracy:           Time taken:
# 1         0.00886749339           89.919998%          92.64282050
# 2         0.00886749339           89.919998%          93.87042083
# 3         0.00886749339           89.9199981%         89.55276787
