from cloud_cover_lstm import *

# Testing LSTM Discriminative ability on test set

test_loss_history = []
test_accuracy_history = []

x_test_seq_cloud_cover, y_test_seq_cloud_cover = create_sequences(x_test_cloud_cover, y_test_cloud_cover, seq_length)

# Convert sequences to PyTorch tensors, similar to train/val
x_test_seq_cloud = to_tensor(x_test_seq_cloud_cover, device_obj, target=False)
y_test_seq_cloud = to_tensor(y_test_seq_cloud_cover, device_obj, target=True)

with torch.no_grad():
  lstm_cloud_cover.eval() # set neural network to evaluation mode
  test_logits = lstm_cloud_cover.forward(x_test_seq_cloud) # Use the sequenced test data
  test_loss = loss_function(test_logits, y_test_seq_cloud) # Use the sequenced test target
  test_loss_history.append(test_loss.item())
  test_accuracy = torch.mean((torch.argmax(test_logits, dim=1) == y_test_seq_cloud).type(torch.float))
  test_accuracy_history.append(test_accuracy.item())

  print(f"Test loss, {test_loss.item():.4f}")
  print(f"Test Accuracy', {test_accuracy.item():.3%}")


  ## ROC Curve for classifier metric
  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plt
  import numpy as np

  # Convert logits to probabilities
  test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
  # Use the sequenced target data for y_test_np to match the length of test_probs
  y_test_np = y_test_seq_cloud.cpu().numpy()

  #  class names
  class_names = ['NCD', 'NSC', 'CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV ']

  # Plot ROC curve for each class
  plt.figure(figsize=(10, 8))

  for i in range(len(class_names)):  # Iterate through all available classes
      if i in np.unique(y_test_np):
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


  ## Confusion Matrix for classifier metric

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  from matplotlib import pyplot as plt
  import numpy as np

  # Use the sequenced target data for y_test_np to match the length of predictions_np
  y_test_np = y_test_seq_cloud.cpu().numpy()
  predictions_np = torch.argmax(test_logits, dim=1).cpu().numpy()

  # Ensure confusion matrix is built for all possible classes, even if some are not present
  cm = confusion_matrix(y_test_np, predictions_np, labels=np.arange(len(class_names)))

  fig, ax = plt.subplots(figsize=(8, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(cmap='Blues', ax=ax, values_format='d')
  plt.title('Confusion Matrix')
  plt.tight_layout()
  plt.show()