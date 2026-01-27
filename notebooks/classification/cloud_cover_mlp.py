import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

from cloud_cover_preprocessing import *

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class CloudCoverNN(nn.Module):
    def __init__(self, input_size=11, output_size=len(sky_cover_order)):
        super(CloudCoverNN, self).__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)


## representation of neural network with input size 10 and an output size of possible categories for cloud cover that can be classified
#  input_size should be the number of features (dimension 1 of x_train_cloud_cover.shape)
cloud_nn = CloudCoverNN(input_size=11, output_size=len(sky_cover_order))
cloud_nn = cloud_nn.to(device)  # Move model to device


def train_model(nn, loss_function, optimizer, epochs, plot=True, progress_bar=True):
    accuracy_history = []
    loss_history = []

    val_accuracy_history = []
    val_loss_history = []
    nn.train()
    epoch_iter = (tqdm(range(epochs)) if progress_bar else range(epochs))

    for epoch in epoch_iter:
        # set all gradients to zero
        optimizer.zero_grad()
        # propagate forwards through network
        prediction = nn.forward(x_train_cloud_cover)
        loss = loss_function(prediction, y_train_cloud_cover)
        loss.backward()  # back propogation to compute gradients
        optimizer.step()  # update gradients
        loss_history.append(loss.item())
        # compute accuracy
        accuracy = torch.mean((torch.argmax(prediction, dim=1) == y_train_cloud_cover).type(torch.float))
        accuracy_history.append(accuracy.item())

        # Validation phase
        nn.eval()  # set model to evaluation mode

        # disable gradient computation

        with torch.no_grad():
            val_logits = nn.forward(x_val_cloud_cover)
            val_loss = loss_function(val_logits, y_val_cloud_cover)
            val_loss_history.append(val_loss.item())
            val_accuracy = torch.mean((torch.argmax(val_logits, dim=1) == y_val_cloud_cover).type(torch.float))
            val_accuracy_history.append(val_accuracy.item())

        if progress_bar:
            epoch_iter.set_postfix({
                'train_loss': f'{loss.item():.4f}',
                'val_loss': f'{val_loss.item():.4f}',
                'train_accuracy': f'{accuracy.item():.4f}',
                'val_accuracy': f'{val_accuracy.item():.4f}'
            })

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        # plot loss
        ax[0].plot(loss_history, label='Training Loss')
        ax[0].plot(val_loss_history, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')

        ax[1].plot(accuracy_history, label='Training Accuracy')
        ax[1].plot(val_accuracy_history, label='Validation Accuracy')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        plt.legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


optimizer = torch.optim.Adam(cloud_nn.parameters(), lr=0.001, weight_decay=1e-5)
loss_function = torch.nn.CrossEntropyLoss()

train_model(cloud_nn, loss_function, optimizer, epochs=200)