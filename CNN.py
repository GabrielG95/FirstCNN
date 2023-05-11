from torch.nn.modules import padding
from torch.optim import optimizer
import torchvision
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from helper_functions import accuracy_fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm.auto import tqdm
from pathlib import Path
import requests
import torch
from torch import nn
import matplotlib.pyplot as plt

# Setup training data
train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None
        )

test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None
        )


# See the first image tensor of our training set
image, label = train_data[0] # image represents one image in our dataset which is a 28x28 tensor


# Let's take a look at our classes (we have 9 classes in total 'images')
class_names = train_data.classes # Will create a list of the names in our class (clothes type)
# print(class_names)

# We'll print out our classes in a directory format so we can see what number represents what image
class_to_idx = train_data.class_to_idx
# print(class_to_idx)

# Setup the batch size hyperparameter
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True) # We shuffle our train data because we don't want our model to learn the order.

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False) # For evaluation purposes it's generally good to have our test data in the same order
                                            # because our model will never actually see the test data set during training.


# time model training
def print_train_time(start: float,
                     end: float,
                     device: torch.device=None):
    total_time = end-start
    print(f'Train time on {device}: {total_time:3f} seconds')
    return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):

    # update as we go through our bathces
    train_loss, train_acc = 0, 0 

    # Loop iterates over the batches in the training dataset.
    for batch, (X, y) in enumerate(data_loader):
        # For each batch, it sets the model to training model.
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%')

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device='cpu'):

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f'\nTest loss: {test_loss:.4f}, Test acc: {test_acc:.2f}')

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

        # Scale the loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {'model_name': model.__class__.__name__,
            'model_loss':loss.item(),
            'model_acc':acc}

class FashionMNISTModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        # Run these functions in a Sequential manner.
        self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*7*7,
                          out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

model_1 = FashionMNISTModel(input_shape=1,
                            hidden_units=10,
                            output_shape=len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

print(model_1)

train_time_start = timer()
epochs = 6
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n---------')
    train_step(model=model_1,
               data_loader=train_dataloader,
               optimizer=optimizer,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)

train_time_end = timer()
eval_model(model=model_1,
           data_loader=test_dataloader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn)
total_train_time = print_train_time(start=train_time_start,
                                    end=train_time_end)















