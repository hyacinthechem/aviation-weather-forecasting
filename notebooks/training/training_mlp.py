
from notebooks.preprocessing.weather_preprocessing import y_test_temp
from notebooks.training.weather_preprocessing import y_train_vsby, y_test_vsby, y_train_temp
from weather_preprocessing import x_train_processed_vsby, x_train_processed_vsby, x_train_processed_temp, x_test_processed_vsby, x_val_processed_vsby, x_train_processed_wind, y_train_wspeed, x_test_processed_temp, x_test_processed_wind, y_test_wspeed

import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

import torch
import pandas as pd

## Convert preprocessed data into tensors

# Wind
x_train_processed_wind = torch.tensor(x_train_processed_wind, dtype=torch.float32)
y_train_processed_wind = torch.tensor(y_train_wspeed, dtype=torch.float32)

x_test_processed_wind = torch.tensor(x_test_processed_wind, dtype=torch.float32)
y_test_processed_wind = pd.DataFrame(y_test_wspeed, columns=['wind'])
y_test_processed_wind = torch.tensor(y_test_processed_wind, dtype=torch.float32)

# Visibility
x_train_processed_vsby = torch.tensor(x_train_processed_vsby, dtype=torch.float32)
y_train_processed_vsby = torch.tensor(y_train_vsby, dtype=torch.float32)

x_test_processed_vsby = torch.tensor(x_test_processed_vsby, dtype=torch.float32)
y_test_processed_vsby = pd.DataFrame(y_test_vsby, columns=['wind'])
y_test_processedvsby = torch.tensor(y_test_processed_vsby, dtype=torch.float32)

# Temperature
x_train_processed_temp = torch.tensor(x_train_processed_temp, dtype=torch.float32)
y_train_processed_temp = torch.tensor(y_train_temp, dtype=torch.float32)

x_test_processed_temp = torch.tensor(x_test_processed_temp, dtype=torch.float32)
y_test_processed_temp = pd.DataFrame(y_test_temp, columns=['wind'])
y_test_processed_temp = torch.tensor(y_test_processed_temp, dtype=torch.float32)


if torch.cuda.is_available():
    x_train_processed_wind = x_train_processed_wind.cuda()
    y_train_processed_wind = y_train_processed_wind.cuda()
    x_test_processed_wind = x_test_processed_wind.cuda()
    y_test_processed_wind = y_test_processed_wind.cuda()

## Define module for pytorch NN
class WeatherNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(WeatherNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


## representation of neural network with 10 input size, 1 output size and 64 hidden size
weather_nn = WeatherNN(input_size=10, output_size=1, hidden_size=64)
print(weather_nn)

"""
Schultz et al.

Found that 200 hidden layers, batch size of 40 with an epoch of 50 paired with adamax optimiser and softmax activation function

Alves et al.

3 layers:
one input
one hidden
one output

linear transfer function used

"""

## method to train model
def train_model(nn, x_train, y_train, loss_function, optimizer, epochs, plot=True, progress_bar=True):
    loss_history = []
    accuracy_history = []
    epoch_iter = ( tqdm(range(epochs)) if progress_bar else range(epochs))

    for epoch in epoch_iter:
        # set all gradients to zero
        optimizer.zero_grad()
        # propagate forwards through network
        logits = nn.forward(x_train)
        loss = loss_function(logits, y_train)
        loss.backward() # back propagation to compute gradients
        optimizer.step() # update gradients

        # compute accuracy
        accuracy = torch.mean(torch.argmax(logits, dim=1) == y_train).float()
        accuracy_history.append(accuracy.item())
        loss_history.append(loss.item())

    if plot:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].plot(loss_history)
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[1].plot(accuracy_history)
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        plt.show()

    return nn, loss_history, accuracy_history


## Training for predicting wind speed

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_wind, y_train_processed_wind, loss_function, optimizer, epochs=50)

# Training for predicting visibility

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_vsby, y_train_processed_vsby, loss_function, optimizer, epochs=50 )

# Training for predicting temperature

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_temp, y_train_processed_temp, loss_function, optimizer, epochs=50)


if __name__ == '__main__':
    pass
