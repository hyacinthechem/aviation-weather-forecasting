import numpy as np
from notebooks.training import training_mlp
from notebooks.training import training_mlp_extended
import matplotlib.pyplot as plt
# track test loss

test_loss = 0.0

mae_history = []
mse_history = []
rmse_history = []
import torch
def test_model(trained_model, x_test, y_test, model_type, target_name):

  # put model in evaluation mode
  trained_model.eval()
  with torch.no_grad():
      predictions = trained_model(x_test)
      mae = torch.mean(torch.abs(predictions - y_test))
      mae_history.append(mae.item())
      mse = torch.mean((predictions - y_test) ** 2)
      mse_history.append(mse.item())
      rmse = torch.sqrt(mse)
      rmse_history.append(rmse.item())

      # calculate residuals
      residuals = predictions - y_test

      # convert to numpy for plotting
      predictions_np = predictions.cpu().numpy().flatten()
      y_test_np = y_test.cpu().numpy().flatten()
      residuals_np = residuals.cpu().numpy().flatten()

      # Print metrics
      print(f'MAE: {mae.item()}')
      print(f'MSE: {mse.item()}')
      print(f'RMSE: {rmse.item()}')

      # Create figure with subplots
      fig, axes = plt.subplots(2,2, figsize=(14,10))
      fig.suptitle(f'{model_type} vs. {target_name} Analysis', fontsize=16, fontweight='bold')

      # predicted vs actual line of best fit
      axes[0,0].scatter(y_test_np, predictions_np, alpha=0.5, s=20)

      # calculate and plot line of best fit
      z = np.polyfit(y_test_np, predictions_np, 1)
      p = np.poly1d(z)
      axes[0, 0].plot(y_test_np, p(y_test_np), "r--", linewidth=2, label=f'Best fit: y={z[0]:.2f}x+{z[1]:.2f}')

      # plot perfect prediction line
      min_val = min(y_test_np.min(), predictions_np.min())
      max_val = max(y_test_np.max(), predictions_np.max())
      axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1, label='Perfect prediction')

      axes[0,0].set_xlabel('Actual Values')
      axes[0,0].set_ylabel('Predicted Values')
      axes[0,0].set_title('Predicted vs. Actual Values')
      axes[0,0].legend()
      axes[0,0].grid(True, alpha=0.3)

      # Residual plot (Residuals vs Predicted)
      axes[0,1].scatter(predictions_np, residuals_np, alpha=0.5, s=20)
      axes[0,1].axhline(0, color='r', linestyle='--', linewidth=2)
      axes[0,1].set_xlabel('Predicted Values')
      axes[0,1].set_ylabel('Residuals ( Predicted - Actual )')
      axes[0,1].set_title('Residual Plot')
      axes[0,1].grid(True, alpha=0.3)

      # Histogram of residuals
      axes[1,0].hist(residuals_np, bins=50, edgecolor='black', alpha=0.7)
      axes[1,0].axvline(0, color='r', linestyle='--', linewidth=2)
      axes[1,0].set_xlabel('Residuals ( Predicted - Actual )')
      axes[1,0].set_ylabel('Frequency')
      axes[1,0].set_title('Distribution of Residuals')
      axes[1,0].grid(True, alpha=0.3)

      from scipy import stats
      stats.probplot(residuals_np, dist='norm', plot=axes[1,1])
      axes[1,1].set_title('Q-Q Plot ( Normality Check )')
      axes[1,1].grid(True, alpha=0.3)

      plt.tight_layout()
      plt.show()

  return predictions_np, y_test_np, residuals_np

from notebooks.training.training_mlp import weather_nn, x_test_processed_wind, y_test_processed_wind, x_test_processed_vsby, y_test_processed_vsby, x_test_processed_temp, y_test_processed_temp

# test base model on predicting wind speed
pred_wind, actual_wind, residual_wind = test_model(
    weather_nn,
    x_test_processed_wind,
    y_test_processed_wind,
    model_type='Base Model',
    target_name='Wind'
)

# test base model on predicting visibility
pred_visibility, actual_visibility, residual_visibility = test_model(
    weather_nn,
    x_test_processed_vsby,
    y_test_processed_vsby,
    model_type='Base Model',
    target_name='Visibility'
)

# test base model on predicting temperature
pred_temp, actual_temp, residual_temp = test_model(
    weather_nn,
    x_test_processed_temp,
    y_test_processed_temp,
    model_type='Base Model',
    target_name='Temperature'
)

from notebooks.training.training_mlp import x_test_processed_wind, y_test_processed_wind, x_test_processed_vsby, y_test_processed_vsby, x_test_processed_temp, y_test_processed_temp
from notebooks.training.training_mlp_extended import weather_nn_extended

# test extended model on predicting wind speed
pred_wind_ext, actual_wind_ext, residual_wind_ext = test_model(
    weather_nn_extended,
    x_test_processed_wind,
    y_test_processed_wind,
    model_type='Extended Model',
    target_name='Wind'
)

# test extended model on predicting visibility
pred_visibility_ext, actual_visibility_ext, residual_visibility_ext = test_model(
    weather_nn_extended,
    x_test_processed_vsby,
    y_test_processed_vsby,
    model_type='Extended Model',
    target_name='Visibility'
)

# test extended model on predicting temperature
pred_temp_ext, actual_temp_ext, residual_temp_ext = test_model(
    weather_nn_extended,
    x_test_processed_temp,
    y_test_processed_temp,
    model_type='Extended Model',
    target_name='Temperature'
)

# Overall metrics comparison between base model and extended model
n_metrics = len(mae_history)
labels = ['Base-Wind', 'Base-Visibility', 'Base-Temperature', 'Base-Visibility', 'Extended-Wind', 'Extended-Visibility', 'Extended-Temperature'][:n_metrics]

fig, axes = plt.subplots(1, 3, figsize=(16,5))
fig.suptitle('Model Error Metrics Comparison', fontsize=16, fontweight='bold')

x_pos = np.arange(len(labels))

#MAE Comparison
axes[0].bar(x_pos, mae_history, color='blue', alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels, rotation=45, ha='right')
axes[0].set_ylabel('Mean Absolute Error')
axes[0].set_title('MAE Comparison')
axes[0].grid(True, alpha=0.3)

#MAE Comparison
axes[0].bar(x_pos, mae_history, color='blue', alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels, rotation=45, ha='right')
axes[0].set_ylabel('Mean Absolute Error')
axes[0].set_title('MAE Comparison')
axes[0].grid(True, alpha=0.3, axis='y')

#MSE Comparison
axes[1].bar(x_pos, mse_history, color='Orange', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels, rotation=45, ha='right')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('MSE Comparison')
axes[1].grid(True, alpha=0.3, axis='y')

#RMSE Comparison
axes[2].bar(x_pos, rmse_history, color='green', alpha=0.7)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels, rotation=45, ha='right')
axes[2].set_ylabel('Root Mean Squared Error')
axes[2].set_title('RMSE Comparison')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

if __name__ == '__main__':
    pass