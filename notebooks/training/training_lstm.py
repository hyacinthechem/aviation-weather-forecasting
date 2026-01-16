from torch import device

from training_mlp import x_train_processed_wind, y_train_processed_wind, x_test_processed_wind, y_test_processed_wind

from training_mlp import x_train_processed_temp, y_train_processed_temp, x_test_processed_temp, y_test_processed_temp

from training_mlp import x_train_processed_vsby, y_train_processed_vsby, x_test_processed_vsby, y_test_processed_vsby

from training_mlp import y_train_wspeed, y_train_temp, y_train_vsby, y_test_vsby
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np


def to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.tensor(data, dtype=torch.float32)
    else:
        tensor = torch.tensor(np.array(data, dtype=np.float32), dtype=torch.float32)

    return tensor.to(device)


## define module for LSTM nn
class LSTMWeather(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWeather, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


def train_lstm(model, train_loader, loss_function, optimizer, epochs, plot=True, progress_bar=True):
    loss_history = []
    mae_history = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device_obj), y.to(device_obj)
            predictions = model.forward(x)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae = torch.mean(torch.abs(predictions - y))
            mae_history.append(mae.item())
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(loss_history)
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[1].plot(mae_history)
        ax[1].set_title('MAE over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        plt.show()

    return model, loss_history, mae_history


## Creating LSTM architecture
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = x_train_processed_wind.shape[1]
num_layers = 3  # 3 hidden layers
hidden_size = 128  # 128 hidden neurons
output_size = 1  # predict one continuous value
dropout = 0.2  # add dropout for regularisation
lstm_weather = LSTMWeather(input_size, hidden_size, num_layers, output_size, dropout).to(device_obj)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm_weather.parameters(), lr=1e-3)

## Create batch sizes
batch_size = 32

x_train_processed_wind_tensor = to_tensor(x_train_processed_wind, device_obj).unsqueeze(1)
y_train_processed_wind = to_tensor(y_train_wspeed, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_wind_tensor, y_train_processed_wind)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## Training for predicting wind speed
model, loss_history, mae_history = train_lstm(lstm_weather, train_loader, loss_function, optimizer, epochs=100,
                                              plot=True, progress_bar=True)

## Training for predicting visibility
x_train_processed_vsby_tensor = to_tensor(x_train_processed_vsby, device_obj).unsqueeze(1)
y_train_processed_vsby_tensor = to_tensor(y_train_vsby, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_vsby_tensor, y_train_processed_vsby_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model, loss_history, mae_history = train_lstm(lstm_weather, train_loader, loss_function, optimizer, epochs=100,
                                              plot=True, progress_bar=True)

# Training for predicting temperature
x_train_processed_temp_tensor = to_tensor(x_train_processed_temp, device_obj).unsqueeze(1)
y_train_processed_temp_tensor = to_tensor(y_train_temp, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_temp_tensor, y_train_processed_temp_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model, loss_history, mae_history = train_lstm(lstm_weather, train_loader, loss_function, optimizer, epochs=100,
                                              plot=True, progress_bar=True)