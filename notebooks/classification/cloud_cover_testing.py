import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from cloud_cover_mlp import *

with torch.no_grad():
  cloud_nn.eval() # set neural network to evaluation mode
  test_logits = cloud_nn.forward(x_test_cloud_cover)
  test_loss = loss_function(test_logits, y_test_cloud_cover)
  test_accuracy = torch.mean((torch.argmax(test_logits, dim=1) == y_test_cloud_cover).type(torch.float))

  print(f"Test loss, {test_loss.item():.4f}")
  print(f"Test Accuracy', {test_accuracy.item():.3%}")


# Performance metrics

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Convert logits to probabilities
test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
y_test_np = y_test_cloud_cover.cpu().numpy()

#  class names
class_names = ['NCD', 'NSC', 'CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV ']

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))

for i in range(6):  # for each of the 6 classes
    y_binary = (y_test_np == i).astype(int)

    y_score = test_probs[:, i]

    # calculate ROC curve
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - One-vs-Rest')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

y_test_np = y_test_cloud_cover.cpu().numpy()
predictions_np = torch.argmax(test_logits, dim=1).cpu().numpy()


cm = confusion_matrix(y_test_np, predictions_np)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()