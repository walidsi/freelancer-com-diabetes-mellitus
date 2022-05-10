import pandas as pd
from sqlalchemy import Integer


def get_float_features(df: pd.DataFrame) -> list:
    return list(df.select_dtypes(include=['float64']).columns)


def get_categorical_features(df: pd.DataFrame) -> list:
    return list(df.select_dtypes(include=['object']).columns)


def get_integer_features(df: pd.DataFrame) -> list:
    integer_list = list(df.select_dtypes(include=['int64']).columns)

    integer_list2 = []
    # remove binary columns
    for column in integer_list:
        value_counts = df[column].value_counts()
        if len(value_counts) > 2:
            integer_list2.append(column)
    return integer_list2


def get_binary_features(df: pd.DataFrame) -> list:
    bool_list = list(df.select_dtypes(include=['bool']).columns)

    columns = df.columns

    for column in columns:
        value_counts = df[column].value_counts()
        if len(value_counts) == 2 and value_counts.index[0] == 0 and value_counts.index[1] == 1:
            bool_list.append(column)

    return bool_list


def remove_highly_correlated_features(features_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Remove features with a correlation greater than the threshold with other feature

    Args:
        features_df (pd.DataFrame): dataframe of numeric (float) features
        threshold (float): correlation threshold
    Returns:
        pd.DataFrame: dataframe with highly correlated features removed
    """

    corr_matrix = features_df.corr()

    size = len(corr_matrix)

    for i in range(size):
        if i == size:
            break
        name = corr_matrix.columns[i]
        for j in range(i + 1, size):
            if j == size:
                break
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                # remove the column j and row j
                corr_matrix.drop(corr_matrix.columns[j], axis=1, inplace=True)
                corr_matrix.drop(corr_matrix.index[j], axis=0, inplace=True)
                size -= 1

    features_pure_df = features_df[corr_matrix.columns]

    return features_pure_df
