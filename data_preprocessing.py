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
