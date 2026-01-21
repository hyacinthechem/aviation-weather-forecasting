from torch import device

from notebooks.preprocessing.weather_preprocessing import y_val_wspeed, x_val_processed_wind, x_val_processed_vsby, \
    y_val_vsby, x_val_processed_temp, y_val_temp
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


def create_sequences(X, y, seq_length=24):
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X = X.values
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


## define module for LSTM nn
class LSTMWeather(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWeather, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


def train_lstm(model, train_loader, val_loader, loss_function, optimizer, epochs, model_save_path=None,
               patience=15, plot=True, progress_bar=True):
    loss_history = []
    val_loss_history = []
    mae_history = []
    val_mae_history = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        total_mae = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device_obj), y.to(device_obj)
            predictions = model.forward(x)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mae = torch.mean(torch.abs(predictions - y))
            total_mae += mae.item()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        average_mae = total_mae / len(train_loader)
        loss_history.append(average_loss)
        mae_history.append(average_mae)

        model.eval()
        val_total_loss = 0
        val_total_mae = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device_obj), y.to(device_obj)
                predictions = model.forward(x)
                loss = loss_function(predictions, y)
                mae = torch.mean(torch.abs(predictions - y))
                val_total_mae += mae.item()
                val_total_loss += loss.item()

        average_val_loss = val_total_loss / len(val_loader)
        average_val_mae = val_total_mae / len(val_loader)
        val_loss_history.append(average_val_loss)
        val_mae_history.append(average_val_mae)

        if progress_bar:
            print(
                f'Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}, MAE: {average_mae:.4f}, Val MAE: {average_val_mae:.4f}')

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].plot(loss_history, label='Training loss')
        ax[0].plot(val_loss_history, label='Validation loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(mae_history, label='Training MAE')
        ax[1].plot(val_mae_history, label='Validation MAE')
        ax[1].set_title('MAE over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('MAE')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return model, loss_history, val_loss_history, mae_history, val_mae_history


## Creating LSTM architecture
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 3  # 3 hidden layers
hidden_size = 128  # 128 hidden neurons
output_size = 1  # predict one continuous value
dropout = 0.2  # add dropout for regularisation
seq_length = 24

## Create batch sizes
batch_size = 32

x_train_seq_wind, y_train_seq_wind = create_sequences(x_train_processed_wind, y_train_wspeed, seq_length)
x_val_seq_wind, y_val_seq_wind = create_sequences(x_val_processed_wind, y_val_wspeed, seq_length)

x_train_processed_wind_tensor = to_tensor(x_train_seq_wind, device_obj)
y_train_processed_wind = to_tensor(y_train_seq_wind, device_obj).reshape(-1, 1)
x_val_processed_wind_tensor = to_tensor(x_val_seq_wind, device_obj)
y_val_processed_wind = to_tensor(y_val_seq_wind, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_wind_tensor, y_train_processed_wind)
val_dataset = TensorDataset(x_val_processed_wind_tensor, y_val_processed_wind)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

## Training for predicting wind speed
input_size_wind = x_train_processed_wind.shape[1]
lstm_weather_wind = LSTMWeather(input_size_wind, hidden_size, num_layers, output_size, dropout).to(device_obj)
loss_function = torch.nn.MSELoss(reduction='mean')
optimizer_wind = torch.optim.Adam(lstm_weather_wind.parameters(), lr=1e-3)

model_wind, loss_history_wind, val_loss_history_wind, mae_history_wind, val_mae_history_wind = train_lstm(
    lstm_weather_wind,
    train_loader,
    val_loader,
    loss_function,
    optimizer_wind,
    epochs=100,
    model_save_path='model_wind.pth',
    patience=15,
    plot=True,
    progress_bar=True
)

## Training for predicting visibility
x_train_seq_vsby, y_train_seq_vsby = create_sequences(x_train_processed_vsby, y_train_vsby, seq_length)
x_val_seq_vsby, y_val_seq_vsby = create_sequences(x_val_processed_vsby, y_val_vsby, seq_length)

x_train_processed_vsby_tensor = to_tensor(x_train_seq_vsby, device_obj)
y_train_processed_vsby_tensor = to_tensor(y_train_seq_vsby, device_obj).reshape(-1, 1)
x_val_processed_vsby_tensor = to_tensor(x_val_seq_vsby, device_obj)
y_val_processed_vsby_tensor = to_tensor(y_val_seq_vsby, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_vsby_tensor, y_train_processed_vsby_tensor)
val_dataset = TensorDataset(x_val_processed_vsby_tensor, y_val_processed_vsby_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size_vsby = x_train_processed_vsby.shape[1]
lstm_weather_vsby = LSTMWeather(input_size_vsby, hidden_size, num_layers, output_size, dropout).to(device_obj)
optimizer_vsby = torch.optim.Adam(lstm_weather_vsby.parameters(), lr=1e-3)

model_vsby, loss_history_vsby, val_loss_history_vsby, mae_history_vsby, val_mae_history_vsby = (
    train_lstm(
        lstm_weather_vsby,
        train_loader,
        val_loader,
        loss_function,
        optimizer_vsby,
        epochs=100,
        model_save_path='model_vsby.pth',
        patience=15,
        plot=True,
        progress_bar=True)
)

# Training for predicting temperature
x_train_seq_temp, y_train_seq_temp = create_sequences(x_train_processed_temp, y_train_temp, seq_length)
x_val_seq_temp, y_val_seq_temp = create_sequences(x_val_processed_temp, y_val_temp, seq_length)

x_train_processed_temp_tensor = to_tensor(x_train_seq_temp, device_obj)
y_train_processed_temp_tensor = to_tensor(y_train_seq_temp, device_obj).reshape(-1, 1)
x_val_processed_temp_tensor = to_tensor(x_val_seq_temp, device_obj)
y_val_processed_temp_tensor = to_tensor(y_val_seq_temp, device_obj).reshape(-1, 1)

train_dataset = TensorDataset(x_train_processed_temp_tensor, y_train_processed_temp_tensor)
val_dataset = TensorDataset(x_val_processed_temp_tensor, y_val_processed_temp_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size_temp = x_train_processed_temp.shape[1]
lstm_weather_temp = LSTMWeather(input_size_temp, hidden_size, num_layers, output_size, dropout).to(device_obj)
optimizer_temp = torch.optim.Adam(lstm_weather_temp.parameters(), lr=1e-3)

model_temp, loss_history_temp, val_loss_history_temp, mae_history_temp, val_mae_history_temp = (
    train_lstm(lstm_weather_temp,
               train_loader,
               val_loader,
               loss_function,
               optimizer_temp,
               epochs=100,
               model_save_path='model_temp.pth',
               patience=15,
               plot=True,
               progress_bar=True
               ))

if __name__ == '__main__':
    pass