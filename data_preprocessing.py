import numpy as np
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


def numpy_arr_to_dataframe(nparray: np.ndarray, features: list) -> pd.DataFrame:
    """Convert numpy array to dataframe

    Args:
        nparray (np.ndarray): input array
        features: list of features names

    Returns:
        pd.DataFrame: dataframe
    """
    # Create list of column names with the format "colN" (from 1 to N)
    col_names = features
    # Declare pandas.DataFrame object
    df = pd.DataFrame(data=nparray, columns=col_names)
    return df


def discretize_continuous_features(df: pd.DataFrame, continuous_features: list) -> pd.DataFrame:
    """Discretize continuous features
    
    Args: 
        df (pd.DataFrame): input dataframe
        continuous_features (list): list of continuous features
        
    Returns:
        pd.DataFrame: dataframe with continuous features replaced with discretized features
    """
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    discretizer.fit(df[continuous_features])

    arr = discretizer.transform(df[continuous_features])
    arr_df = numpy_arr_to_dataframe(arr)
    arr_df.reset_index(inplace=True, drop=True)
    df.drop(continuous_features, axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = pd.concat([df, arr_df], axis=1)

    return df


def fill_empty_with_average(df: pd.DataFrame, column: str, criteria_column: str) -> pd.DataFrame:
    """Fill empty cells in a column with the average value of the column in the same group

        Args:
            df (pd.DataFrame): input dataframe
            column (str): column to fill
            criteria_column (str): column to group by

        Returns:
            pd.DataFrame: updated dataframe
        """
    # Group by criteria_column and get the mean of column
    if df[column].isnull().sum() == 0:
        return df

    values = df[criteria_column].unique()
    for v in values:
        mask = (df[criteria_column] == v)
        mean = df[mask][column].mean()
        print(f"Mean {v} {column} is {mean}")
        df.loc[mask & (df[column].isnull()), column] = mean

    return df


def remove_outliers(df: pd.DataFrame, columns: list, clip: bool = False) -> pd.DataFrame:
    """Removes rows with outliers from numeric floating point columns

    Args:
        df (pd.DataFrame): input dataframe
        columns (list): numeric (float) columns
        clip (bool): clip values to min/max if True

    Returns:
        pd.DataFrame: updated dataframe
    """
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        maximum = (q3 - q1) * 1.5 + q3
        minimum = (q3 - q1) * -1.5 + q1

        if clip:
            df[column] = df[column].clip(minimum, maximum)
        else:
            df = df[(df[column] <= maximum) & (df[column] >= minimum)]

    return df
