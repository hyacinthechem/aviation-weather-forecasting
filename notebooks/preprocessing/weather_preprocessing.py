import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
"""
Preprocessing techniques:
Imputation methods:
   deletion approach for missing values

Standardisation
"""

apt_data = pd.read_csv("NZWN.csv", na_values=['M'], low_memory=False)
# convert valid to datetime
apt_data['valid'] = pd.to_datetime(apt_data['valid'])
apt_data.head()
apt_data.isnull().sum()


## Split data into training, validation and test sets

"""
Training: data spanning from 2020 to 2023
Validation: data in the year of 2024
Testing: test on data in 2025
"""

train = apt_data[apt_data['valid'].dt.year <= 2023]
val = apt_data[apt_data['valid'].dt.year == 2024]
test = apt_data[apt_data['valid'].dt.year == 2025]

"""
Columns to delete

Station: all entries refer to NZWN
Mean Sea Level Pressure: all missing values
Metar: raw data is already converted for us
Sky conditions: except one column the rest have mostly missing values
Sky Level: except one column the rest have mostly missing values
wxcodes: all missing values
ice accrediation: all missing values
peak wind gust, peak wind direction and peak wind time: all missing values
snowdepth: all missing values


"""

train = train.drop(columns = ['station', 'skyc2', 'skyc3', 'skyc4', 'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'gust','peak_wind_drct', 'peak_wind_gust', 'peak_wind_time', 'snowdepth','mslp', 'metar'])

val = val.drop(columns = ['station', 'skyc2', 'skyc3', 'skyc4', 'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'gust', 'peak_wind_drct', 'peak_wind_gust', 'peak_wind_time', 'snowdepth','mslp', 'metar'])

test = test.drop(columns = ['station', 'skyc2', 'skyc3', 'skyc4', 'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'gust', 'peak_wind_drct', 'peak_wind_gust', 'peak_wind_time', 'snowdepth','mslp', 'metar'])

## utilise deletion approaches for small amounts of missing values except for sky cover where we use imputation approach
train = train.dropna(subset=['tmpf','dwpf', 'relh', 'vsby', 'feel', 'skyc1'])
val = val.dropna(subset=['tmpf','dwpf', 'relh', 'vsby', 'feel','skyc1'])
test = test.dropna(subset=['tmpf','dwpf', 'relh', 'vsby', 'feel','skyc1'])


"""
Fix:
Change from a nowcasting model to a forecasting model
Shifts target columns up by one to use features in previous
columns to predict target at 30 minute intervals ( predicts 30 minutes ahead t+1 ).

"""
train['sknt-target'] = train['sknt'].shift(-1)
train['vsby-target'] = train['vsby'].shift(-1)
train['tmpf-target'] = train['tmpf'].shift(-1)

val['sknt-target'] = val['sknt'].shift(-1)
val['vsby-target'] = val['vsby'].shift(-1)
val['tmpf-target'] = val['tmpf'].shift(-1)

test['sknt-target'] = test['sknt'].shift(-1)
test['vsby-target'] = test['vsby'].shift(-1)
test['tmpf-target'] = test['tmpf'].shift(-1)

## Need to delete rows with NaN values for rows at the bottom that don't have targets after shift occurs
train = train.dropna(subset=['sknt-target', 'vsby-target', 'tmpf-target'])
val = val.dropna(subset=['sknt-target', 'vsby-target', 'tmpf-target'])
test = test.dropna(subset=['sknt-target', 'vsby-target', 'tmpf-target'])

train.head()

def train_target_selection(train, val, test, target_name):

    # Use the shifted target col
    target_col = f'{target_name}-target'

    # Drop both the current and shifted target col from training
    x_train, y_train = train.drop(columns=[target_name, target_col]), train[target_col]

    x_val, y_val = val.drop(columns=[target_name, target_col]), val[target_col]

    x_test, y_test = test.drop(columns=[target_name, target_col]), test[target_col]

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train_wspeed, y_train_wspeed, x_val_wspeed, y_val_wspeed, x_test_wspeed, y_test_wspeed = train_target_selection(train, val, test, "sknt")

x_train_vsby, y_train_vsby, x_val_vsby, y_val_vsby, x_test_vsby, y_test_vsby = train_target_selection(train, val, test, "vsby")

x_train_temp, y_train_temp, x_val_temp, y_val_temp, x_test_temp, y_test_temp = train_target_selection(train, val, test, "tmpf")

train.isnull().sum()

## Standardise data: fit only on training data
## Utilise column transformer for both categorical and numerical columns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def create_preprocessor(target_name):
    sky_cover_order = ['CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV']
    ordinal_cols = ['skyc1']

    all_numeric = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt','p01i', 'alti', 'vsby', 'feel']
    numerical_missing = ['skyl1']
    numerical_complete = [col for col in all_numeric if col!=target_name]

    # Utilise pipeline to impute then scale
    numerical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ]
    )

     # create preprocessor scaler
    preprocessor = ColumnTransformer(
        transformers=[
        ('numerical_missing', numerical_pipeline, numerical_missing),
        ('numerical_complete', StandardScaler(), numerical_complete),
        ('ordinal_skyc', OrdinalEncoder(categories=[sky_cover_order], handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)])

    return preprocessor


## Standardise data using preprocessor for targets

## Wind target
wind_preprocessor = create_preprocessor('sknt')
x_train_processed_wind = wind_preprocessor.fit_transform(x_train_wspeed)
x_val_processed_wind = wind_preprocessor.transform(x_val_wspeed)
x_test_processed_wind = wind_preprocessor.transform(x_test_wspeed)


## Temperature target
temperature_preprocessor = create_preprocessor('tmpf')
x_train_processed_temp = temperature_preprocessor.fit_transform(x_train_temp)
x_val_processed_temp = temperature_preprocessor.transform(x_val_temp)
x_test_processed_temp = temperature_preprocessor.transform(x_test_temp)

## Visibility target
visibility_preprocessor = create_preprocessor('vsby')
x_train_processed_vsby = visibility_preprocessor.fit_transform(x_train_vsby)
x_val_processed_vsby = visibility_preprocessor.transform(x_val_vsby)
x_test_processed_vsby = visibility_preprocessor.transform(x_test_vsby)

# Feature selection to uncover hidden correlations between features of importance
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_target_regressor(x_train, y_train, x_test, y_test, target_name):
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"RMSE: {target_name, mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score: {target_name,r2_score(y_test, y_pred)}")


print("Evaluation for all features")

## Train for "wind speed" target
train_target_regressor(x_train_processed_wind, y_train_wspeed, x_test_processed_wind, y_test_wspeed,"Wind speed")

## Train for "temperature" target
train_target_regressor(x_train_processed_temp, y_train_temp, x_test_processed_temp, y_test_temp,  "Temperature")

## Train for "visibility" target
train_target_regressor(x_train_processed_vsby, y_train_vsby, x_test_processed_vsby, y_test_vsby,"Visibility")


def select_k_best(x_train, y_train, x_test, preprocessor, target_name):
     select_kbest = SelectKBest(score_func = mutual_info_regression, k=5)

     x_train_kbest = select_kbest.fit_transform(x_train, y_train)

     x_test_kbest = select_kbest.transform(x_test)

     feature_names = preprocessor.get_feature_names_out()
     selected_features = select_kbest.get_support()
     selected_feature_names = feature_names[selected_features].tolist()
     print(f"Top 5 features selected for {target_name}:")
     print(selected_feature_names)
     return select_kbest

select_k_best_wind = select_k_best(x_train_processed_wind, y_train_wspeed, x_test_processed_wind, wind_preprocessor, "Wind speed")

select_k_best_temperature = select_k_best(x_train_processed_temp, y_train_temp, x_test_processed_temp,temperature_preprocessor , "Temperature")

select_k_best_visibility = select_k_best(x_train_processed_vsby, y_train_vsby, x_test_processed_vsby, visibility_preprocessor,"Visibility")


## Apply feature selection to reduce to top 5 features


# Wind - reduce to top 5 features
x_train_processed_wind = select_k_best_wind.transform(x_train_processed_wind)
x_val_processed_wind = select_k_best_wind.transform(x_val_processed_wind)
x_test_processed_wind = select_k_best_wind.transform(x_test_processed_wind)
print(f"Wind features shape (after selection): {x_train_processed_wind.shape}")

# Temperature - reduce to top 5 features
x_train_processed_temp = select_k_best_temperature.transform(x_train_processed_temp)
x_val_processed_temp = select_k_best_temperature.transform(x_val_processed_temp)
x_test_processed_temp = select_k_best_temperature.transform(x_test_processed_temp)
print(f"Temperature features shape (after selection): {x_train_processed_temp.shape}")

# Visibility - reduce to top 5 features
x_train_processed_vsby = select_k_best_visibility.transform(x_train_processed_vsby)
x_val_processed_vsby = select_k_best_visibility.transform(x_val_processed_vsby)
x_test_processed_vsby = select_k_best_visibility.transform(x_test_processed_vsby)
print(f"Visibility features shape (after selection): {x_train_processed_vsby.shape}")


# Re-create full feature sets for visualization (heatmap needs all features)
x_train_wind_full = wind_preprocessor.transform(x_train_wspeed)
x_train_vsby_full = visibility_preprocessor.transform(x_train_vsby)
x_train_temp_full = temperature_preprocessor.transform(x_train_temp)


from feature_mapping_utility import visualise_heatmap
visualise_heatmap(x_train_wind_full, wind_preprocessor, select_k_best_wind, "Wind Speed")

visualise_heatmap(x_train_vsby_full, temperature_preprocessor, select_k_best_temperature, "Temperature")

visualise_heatmap(x_train_temp_full, visibility_preprocessor, select_k_best_visibility, "Visibility")



if __name__ == "__main__":
    pass
