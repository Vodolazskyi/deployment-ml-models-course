from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Individual pre-processing and training functions
# ================================================


def load_data(df_path: str) -> pd.DataFrame:
    # Function loads data for training
    return pd.read_csv(df_path)


def divide_train_test(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[target], test_size=0.1, random_state=0
    )
    return X_train, X_test, y_train, y_test


def extract_cabin_letter(df: pd.DataFrame, var: str) -> pd.Series:
    # captures the first letter
    return df[var].str[0]


def add_missing_indicator(df: pd.DataFrame, var: str) -> pd.Series:
    # function adds a binary missing value indicator
    return np.where(df[var].isna(), 1, 0)


def impute_na(
    df: pd.DataFrame, var: str, replacement: Union[str, int, float] = "Missing"
) -> pd.Series:
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)


def remove_rare_labels(
    df: pd.DataFrame, var: str, frequent_labels: List[str]
) -> pd.Series:
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], "Rare")


def encode_categorical(df: pd.DataFrame, var: str) -> pd.DataFrame:
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
    return df.drop([var], axis=1)


def check_dummy_variables(df: pd.DataFrame, dummy_list: List) -> pd.DataFrame:
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    df = df.copy()
    for col in dummy_list:
        if col not in df.columns:
            df[col] = 0
    return df


def train_scaler(df: pd.DataFrame, output_path: str) -> StandardScaler:
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)


def train_model(df: pd.DataFrame, target: pd.Series, output_path: str) -> None:
    # train and save model
    lr = LogisticRegression()
    lr.fit(df, target)

    # save the model
    joblib.dump(lr, output_path)


def predict(df: pd.DataFrame, model: str) -> pd.Series:
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)
