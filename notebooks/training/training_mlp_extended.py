from notebooks.preprocessing.weather_preprocessing import y_test_temp
from notebooks.training.weather_preprocessing import y_train_vsby, y_test_vsby, y_train_temp
from weather_preprocessing import x_train_processed_vsby, x_train_processed_vsby, x_train_processed_temp, x_test_processed_vsby, x_val_processed_vsby, x_train_processed_wind, y_train_wspeed, x_test_processed_temp, x_test_processed_wind, y_test_wspeed
from weather_preprocessing import x_val_processed_wind, x_val_processed_vsby, x_val_processed_temp, y_val_wspeed, y_val_vsby, y_val_temp

import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

## Helper function to convert data to tensor safely
def to_tensor(data, device):
    """Convert data to tensor, handling various input types"""
    if isinstance(data, torch.Tensor):
        tensor = data.clone().detach().float()
    elif isinstance(data, pd.DataFrame):
        tensor = torch.tensor(data.values.astype(np.float32), dtype=torch.float32)
    elif isinstance(data, pd.Series):
        tensor = torch.tensor(data.values.astype(np.float32), dtype=torch.float32)
    elif isinstance(data, np.ndarray):
        tensor = torch.tensor(data.astype(np.float32), dtype=torch.float32)
    else:
        tensor = torch.tensor(np.array(data, dtype=np.float32), dtype=torch.float32)

    return tensor.to(device)


## Convert preprocessed data into tensors

# Wind
x_train_processed_wind = to_tensor(x_train_processed_wind, device)
y_train_processed_wind = to_tensor(y_train_wspeed, device).reshape(-1, 1)  # Reshape to [N, 1]
x_val_processed_wind = to_tensor(x_val_processed_wind, device)
y_val_processed_wind = to_tensor(y_val_wspeed, device).reshape(-1, 1)  # Reshape to [N, 1]

# Visibility
x_train_processed_vsby = to_tensor(x_train_processed_vsby, device)
y_train_processed_vsby = to_tensor(y_train_vsby, device).reshape(-1, 1)  # Reshape to [N, 1]
x_val_processed_vsby = to_tensor(x_val_processed_vsby, device)
y_val_processed_vsby = to_tensor(y_val_vsby, device).reshape(-1, 1)  # Reshape to [N, 1]

# Temperature
x_train_processed_temp = to_tensor(x_train_processed_temp, device)
y_train_processed_temp = to_tensor(y_train_temp, device).reshape(-1, 1)  # Reshape to [N, 1]
x_val_processed_temp = to_tensor(x_val_processed_temp, device)
y_val_processed_temp = to_tensor(y_val_temp, device).reshape(-1, 1)  # Reshape to [N, 1]


"""
Takes 10 features as input:
Expands into 64 features in hidden layer

Second layer compresses from 64 down to 32. network is being forced to distill information. Keeps
only what is useful for prediction. uses ReLU activation function

Third Layer further compresses from 32 down to 16. uses ReLU activation function

Output layers takes those 16  values and produces a single output.

"""
## Define module for pytorch NN
class WeatherNN(torch.nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=64):
        super(WeatherNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)  # 64 → 32
        self.fc3 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)  # 32 → 16
        self.fc4 = torch.nn.Linear(hidden_size // 4, output_size)  # 16 → 1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output for regression
        return x

## representation of neural network with 10 input size, 1 output size and 64 hidden size
weather_nn = WeatherNN(input_size=10, output_size=1, hidden_size=64)
weather_nn = weather_nn.to(device)  # Move model to device
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
def train_model(nn, x_train, y_train, x_val, y_val, loss_function, optimizer, epochs, plot=True, progress_bar=True):
    loss_history = []
    mae_history = []
    val_loss_history = []
    val_mae_history = []
    epoch_iter = (tqdm(range(epochs)) if progress_bar else range(epochs))

    for epoch in epoch_iter:
        # Training phase
        nn.train()
        # set all gradients to zero
        optimizer.zero_grad()
        # propagate forwards through network
        logits = nn.forward(x_train)
        loss = loss_function(logits, y_train)
        loss.backward() # back propagation to compute gradients
        optimizer.step() # update gradients

        # compute accuracy
        mae = torch.mean(torch.abs(logits-y_train)).item()
        mae_history.append(mae)
        loss_history.append(loss.item())

        # Validation Phase
        nn.eval() # set model to evaluation mode
        # disable gradient computation
        with torch.no_grad():
            val_logits = nn.forward(x_val)
            val_loss = loss_function(val_logits, y_val)
            val_mae = torch.mean(torch.abs(val_logits-y_val)).item()
            val_loss_history.append(val_loss.item())
            val_mae_history.append(val_mae)

        # update progress bar with metrics
        if progress_bar:
            epoch_iter.set_postfix({
                'train_loss' : f'{loss.item():.4f}',
                'val_loss' : f'{val_loss.item():.4f}',
                'train_mae' : f'{mae:.4f}',
                'val_mae' : f'{val_mae:.4f}'
            })


    if plot:
        fig, ax = plt.subplots(1,2, figsize=(14,5))

        # plot loss
        ax[0].plot(loss_history, label='Training Loss')
        ax[0].plot(val_loss_history, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')

        ax[1].plot(mae_history, label='Training MAE')
        ax[1].plot(val_mae_history, label='Validation MAE')
        ax[1].set_title('MAE over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('MAE')
        plt.legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    return nn, loss_history, val_loss_history, mae_history,  val_mae_history


## Training for predicting wind speed

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_wind, y_train_processed_wind, x_val_processed_wind, y_val_processed_wind, loss_function, optimizer, epochs=200)

# Training for predicting visibility

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_vsby, y_train_processed_vsby, x_val_processed_vsby, y_val_processed_vsby, loss_function, optimizer, epochs=200)

# Training for predicting temperature

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(weather_nn.parameters(), lr=0.001)

model, loss_history, accuracy_history = train_model(weather_nn, x_train_processed_temp, y_train_processed_temp, x_val_processed_temp, y_val_processed_temp, optimizer, epochs=200)


if __name__ == '__main__':
    pass
