import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_df(df: pd.DataFrame, test_column: str) -> pd.DataFrame:
    df = df.loc[:, df.nunique(dropna=False)!=1]
    df = df.drop_duplicates()
    df = df.T.drop_duplicates().T
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    # remove rows where test column is Nan
    df = df[~df[test_column].isna()]


    # drop features with identical values for all samples
    #df = df.loc[:, ~df.nunique(dropna=False)==1]
    return df


def preprocess(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Choose only numeric columns
    X = X.select_dtypes(include=['number'])
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # assert no string columns in X
    assert len(X_test.select_dtypes(include=['object']).columns)==0

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Replace NaN with 0
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_test, y_train, y_test


def get_scattered_chunks(
    data: pd.DataFrame, n_chunks: int = 5, chunk_size: int = 3
) -> pd.DataFrame:
    """
    Returns a subsample of equally scattered chunks of rows.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    n_chunks : int
        Number of chunks to collect for the subsample
    chunk_size : int
        Number of rows to include in each chunk

    Returns
    -------
    pd.DataFrame
        Subsample data
    """

    endpoint = len(data) - chunk_size
    sample_indices = np.linspace(0, endpoint, n_chunks, dtype=int)
    sample_indices = [
        index for i in sample_indices for index in range(i, i + chunk_size)
    ]
    return data.iloc[sample_indices, :]