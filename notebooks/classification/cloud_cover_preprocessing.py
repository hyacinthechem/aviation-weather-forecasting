import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load dataset
apt_data = pd.read_csv("NZWN.csv", na_values=['M'], low_memory=False)

# convert valid to datetime

apt_data['valid'] = pd.to_datetime(apt_data['valid'])
apt_data.head()

# statistics on null values
apt_data.isnull().sum()

# split training, validation and test sets

train = apt_data[apt_data['valid'].dt.year <=2023]
val = apt_data[apt_data['valid'].dt.year == 2024]
test = apt_data[apt_data['valid'].dt.year == 2025]

# drop columns that have mostly missing values

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

train.head()


def create_preprocessor(target_name):
    sky_cover_order = ['CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV']
    ordinal_cols = ['skyc1']

    all_numeric = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti', 'vsby', 'feel']
    numerical_missing = ['skyl1']
    numerical_complete = [col for col in all_numeric if col != target_name]

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
            ('ordinal_skyc',
             OrdinalEncoder(categories=[sky_cover_order], handle_unknown='use_encoded_value', unknown_value=-1),
             ordinal_cols)])

    return preprocessor

# Standardise data using preprocessor for target

## Cloud cover target
cloud_cover_preprocessor = create_preprocessor('skyc1')
x_train_cloud_cover = cloud_cover_preprocessor.fit_transform(train)
x_val_cloud_cover = cloud_cover_preprocessor.transform(val)
x_test_cloud_cover = cloud_cover_preprocessor.transform(test)

## data to predict
y_train_cloud_cover = train['skyc1']
y_val_cloud_cover = val['skyc1']
y_test_cloud_cover = test['skyc1']

if __name__ == '__main__':
    pass
