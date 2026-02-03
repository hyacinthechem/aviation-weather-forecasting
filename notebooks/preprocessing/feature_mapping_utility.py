"""
Feature Name Mapping Utility
Use this in any of your preprocessing scripts to get human-readable feature names.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FEATURE_NAME_MAP = {
    # Numerical features (complete - no missing values)
    'numerical_complete__tmpf': 'Temperature (°F)',
    'numerical_complete__dwpf': 'Dew Point (°F)',
    'numerical_complete__relh': 'Relative Humidity (%)',
    'numerical_complete__drct': 'Wind Direction (°)',
    'numerical_complete__sknt': 'Wind Speed (knots)',
    'numerical_complete__p01i': 'Precipitation (in)',
    'numerical_complete__alti': 'Altimeter (inHg)',
    'numerical_complete__vsby': 'Visibility (mi)',
    'numerical_complete__feel': 'Feels Like (°F)',

    # Numerical features (with missing values - imputed)
    'numerical_missing__skyl1': 'Sky Level 1 (ft)',

    # Ordinal/categorical features
    'ordinal_skyc__skyc1': 'Sky Cover',
}


def get_readable_name(technical_name):

    return FEATURE_NAME_MAP.get(technical_name, technical_name)


def get_readable_names(technical_names):
    return [get_readable_name(name) for name in technical_names]


def visualise_heatmap(x_train_processed, preprocessor, selector, target_name):
    """
    Visualize correlation heatmap of selected features with readable names.
"""
    # Get selected feature names
    feature_names = preprocessor.get_feature_names_out()
    selected_feature_names = feature_names[selector.get_support()].tolist()

    # Get indices of selected features
    selected_indices = selector.get_support()

    # Extract selected features from numpy array and convert to DataFrame
    selected_features_data = pd.DataFrame(
        x_train_processed[:, selected_indices],
        columns=selected_feature_names
    )

    # Rename columns to human-readable names
    readable_names = get_readable_names(selected_feature_names)
    selected_features_data.columns = readable_names

    # Calculate correlation matrix
    correlation_matrix = selected_features_data.corr(method='pearson')

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, fmt='.2f', linewidths=.5, square=True,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title(f'Feature Correlation Heatmap: {target_name}',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()


def print_selected_features(preprocessor, selector, target_name):
    """
    Print selected features with readable names.

    Parameters:
    - preprocessor: fitted preprocessor
    - selector: fitted SelectKBest selector
    - target_name: name of target for printing
    """
    feature_names = preprocessor.get_feature_names_out()
    selected_features = selector.get_support()
    selected_feature_names = feature_names[selected_features].tolist()

    print(f"\nTop {selector.k} features selected for {target_name}:")
    for i, tech_name in enumerate(selected_feature_names, 1):
        readable = get_readable_name(tech_name)
        print(f"  {i}. {readable}")
        if readable != tech_name:  # Show technical name if different
            print(f"     ({tech_name})")


# Example usage:
if __name__ == '__main__':
 pass