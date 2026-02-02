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

"""
Shifts target columns up by one to use features in previous
columns to predict target at 30 minute intervals ( predicts 30 minutes ahead t+1 )

"""

train['skyc1-target'] = train['skyc1'].shift(-1)
val['skyc1-target'] = val['skyc1'].shift(-1)
test['skyc1-target'] = test['skyc1'].shift(-1)

train = train.dropna(subset=['skyc1-target'])
val = val.dropna(subset=['skyc1-target'])
test = test.dropna(subset=['skyc1-target'])


train.head()

"""
Method that drops target from column. Model was originally training on cloud cover target 
which was giving incorrect performance results
"""
def train_target_selector(train, val, test, target):
    target_col = f'{target}-target'
    x_train, y_train = train.drop(columns=[target, target_col]), train[target_col]
    x_val, y_val = val.drop(columns=[target, target_col]), val[target_col]
    x_test, y_test = test.drop(columns=[target, target_col]), test[target_col]
    return x_train, y_train, x_val, y_val, x_test, y_test

(x_train_cloud_cover,
 y_train_cloud_cover,
 x_val_cloud_cover, y_val_cloud_cover,
 x_test_cloud_cover, y_test_cloud_cover) = train_target_selector(train, val, test, 'skyc1')

def create_preprocessor(target_name):
    sky_cover_order = ['CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'VV']
    ordinal_cols = ['skyc1'] if target_name!='skyc1' else []
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

    transformers=[
            ('numerical_missing', numerical_pipeline, numerical_missing),
            ('numerical_complete', StandardScaler(), numerical_complete),
            ]

    if ordinal_cols:
        transformers.append(('ordinal_skyc',
        OrdinalEncoder(categories=[sky_cover_order], handle_unknown='use_encoded_value', unknown_value=-1),
        ordinal_cols))



    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

# Standardise data using preprocessor for target

## Cloud cover target
cloud_cover_preprocessor = create_preprocessor('skyc1')
x_train_cloud_cover = cloud_cover_preprocessor.fit_transform(x_train_cloud_cover)
x_val_cloud_cover = cloud_cover_preprocessor.transform(x_val_cloud_cover)
x_test_cloud_cover = cloud_cover_preprocessor.transform(x_test_cloud_cover)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
def train_target_classifier(x_train, y_train, x_test, y_test, target_name):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Accuracy for features {target_name, balanced_accuracy_score(y_test, y_pred) }')
    return model


print("Evaluation for Base Line model")

baseline_model = train_target_classifier(
    x_train_cloud_cover,
    y_train_cloud_cover,
    x_test_cloud_cover,
    y_test_cloud_cover,
    'skyc1')


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from functools import partial
def select_k_best(x_train, y_train, x_test, preprocessor, target_name):
    mutual_info_fix = partial(mutual_info_classif, random_state=231)
    selector_kbest = SelectKBest(score_func=mutual_info_fix, k=5)
    x_train_best = selector_kbest.fit_transform(x_train, y_train)
    x_test_best = selector_kbest.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    selected_features = selector_kbest.get_support()
    selected_feature_names = feature_names[selected_features].tolist()
    print(f"Top 5 features selected for {target_name}:")
    print(selected_feature_names)
    return selector_kbest

cloud_cover_selector = select_k_best(x_train_cloud_cover, y_train_cloud_cover, x_test_cloud_cover, cloud_cover_preprocessor, 'skyc1')

## Apply feature selection transformation to new training set to reduce to five features
x_train_cloud_cover_selected = cloud_cover_selector.transform(x_train_cloud_cover)
x_val_cloud_cover_selected = cloud_cover_selector.transform(x_val_cloud_cover)
x_test_cloud_cover_selected = cloud_cover_selector.transform(x_test_cloud_cover)


print("Evaluation for Feature Selection")
selected_model = train_target_classifier(
    x_train_cloud_cover_selected,
    y_train_cloud_cover,x_test_cloud_cover_selected,
    y_test_cloud_cover,
    'skyc1')



from feature_mapping_utility import visualise_heatmap, print_selected_features
visualise_heatmap(
    x_train_cloud_cover,
    cloud_cover_preprocessor,
    cloud_cover_selector,
    'Cloud Cover')


if __name__ == '__main__':
    pass
