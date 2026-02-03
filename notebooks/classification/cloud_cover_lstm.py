import numpy as np
import matplotlib.pyplot as plt
import torch

from cloud_cover_preprocessing import *
from torch import nn
from torch import optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_tensor(data, device, target=False):
    """Convert data to tensor, handling various input types
       Data can be in the form of a pandas series, dataframe or numpy array

       if the target is true ( classify cloud cover ) pytorch requires tensor dtype
       to be of long for classification tasks
    """

    if target:
        dtype = torch.long
    else:
        dtype = torch.float32

    if isinstance(data, torch.Tensor):
        tensor = data.clone().detach().to(dtype=dtype)
    elif isinstance(data, pd.DataFrame):
        tensor = torch.tensor(data.values.astype(np.float32), dtype=dtype)
    elif isinstance(data, pd.Series):
        tensor = torch.tensor(data.values, dtype=dtype)
    elif isinstance(data, np.ndarray):
        tensor = torch.tensor(data, dtype=dtype)
    else:
        tensor = torch.tensor(np.array(data, dtype=np.float32), dtype=dtype)

    return tensor.to(device)


x_train_cloud_cover = to_tensor(x_train_cloud_cover, device)
x_val_cloud_cover = to_tensor(x_val_cloud_cover, device)
x_test_cloud_cover = to_tensor(x_test_cloud_cover, device)

sky_cover_order = ['NCD', 'NSC', 'CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV ']
label_encoder = OrdinalEncoder(categories=[sky_cover_order], handle_unknown='use_encoded_value', unknown_value=-1)

y_train_series = train['skyc1']
y_val_series = val['skyc1']
y_test_series = test['skyc1']

# Apply the encoder to the series (converted to numpy array for reshape)
y_train_cloud_cover_encoded = label_encoder.fit_transform(y_train_series.to_numpy().reshape(-1, 1)).flatten()
y_val_cloud_cover_encoded = label_encoder.transform(y_val_series.to_numpy().reshape(-1, 1)).flatten()
y_test_cloud_cover_encoded = label_encoder.transform(y_test_series.to_numpy().reshape(-1, 1)).flatten()

# Enable target as true to convert dtypes of encoded value to torch.long
# which is what is required for pytorch classification
y_train_cloud_cover = to_tensor(y_train_cloud_cover_encoded, device, target=True)
y_val_cloud_cover = to_tensor(y_val_cloud_cover_encoded, device, target=True)
y_test_cloud_cover = to_tensor(y_test_cloud_cover_encoded, device, target=True)


def create_sequences(X, y, seq_length=24):
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Move tensors to CPU before converting to numpy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


class CloudCoverLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CloudCoverLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs, model_save_path=None, patience=15,
                plot=True, progress_bar=True):
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device_obj), y.to(device_obj)
            predictions = model.forward(x)
            loss = loss_function(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == y).sum().item()
            total_samples += y.size(0)

        average_loss = total_loss / len(train_loader)
        average_accuracy = total_correct / total_samples if total_samples > 0 else 0
        loss_history.append(average_loss)
        accuracy_history.append(average_accuracy)

        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device_obj), y.to(device_obj)
                predictions = model.forward(x)
                loss = loss_function(predictions, y)

                val_total_loss += loss.item()

                predicted_labels = torch.argmax(predictions, dim=1)
                val_total_correct += (predicted_labels == y).sum().item()
                val_total_samples += y.size(0)

        average_val_loss = val_total_loss / len(val_loader)
        average_val_accuracy = val_total_correct / val_total_samples if val_total_samples > 0 else 0
        val_loss_history.append(average_val_loss)
        val_accuracy_history.append(average_val_accuracy)

        if progress_bar:
            print(
                f'Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}, Accuracy: {average_accuracy:.4f}, Val Accuracy: {average_val_accuracy:.4f}')

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

        ax[1].plot(accuracy_history, label='Training Accuracy')
        ax[1].plot(val_accuracy_history, label='Validation Accuracy')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return model, loss_history, val_loss_history, accuracy_history, val_accuracy_history


# Creating LSTM Architecture

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device_obj = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_layers = 3  # 3 hidden layers
hidden_size = 128  # 128 hidden neurons
output_size = len(sky_cover_order)  ## the amount of classes to classify
dropout = 0.2  # add dropout for regularisation
seq_length = 24

# Create batch sizes
batch_size = 32

x_train_seq_cloud_cover, y_train_seq_cloud_cover = create_sequences(x_train_cloud_cover, y_train_cloud_cover,
                                                                    seq_length)
x_val_seq_cloud_cover, y_val_seq_cloud_cover = create_sequences(x_val_cloud_cover, y_val_cloud_cover, seq_length)

x_train_seq_cloud = to_tensor(x_train_seq_cloud_cover, device_obj, target=False)
y_train_seq_cloud = to_tensor(y_train_seq_cloud_cover, device_obj, target=True)
x_val_seq_cloud = to_tensor(x_val_seq_cloud_cover, device_obj, target=False)
y_val_seq_cloud = to_tensor(y_val_seq_cloud_cover, device_obj, target=True)

train_dataset = TensorDataset(x_train_seq_cloud, y_train_seq_cloud)
val_dataset = TensorDataset(x_val_seq_cloud, y_val_seq_cloud)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Training for predicting cloud cover

input_size = x_train_seq_cloud.shape[2] if len(x_train_seq_cloud.shape) > 2 else x_train_seq_cloud.shape[1]
lstm_cloud_cover = CloudCoverLSTM(input_size, hidden_size, num_layers, output_size, dropout).to(device_obj)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_cloud_cover.parameters(), lr=0.001)

model_cloud_cover, loss_history_cloud, val_loss_cloud, accuracy_cloud, val_accuracy_cloud = train_model(
    lstm_cloud_cover,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    epochs=100,
    model_save_path='cloud_model.pth',
    patience=15,
    plot=True,
    progress_bar=True
)