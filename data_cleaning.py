import pandas as pd

def filter_df_dtypes(df, include=None, exclude=None):
    """Function that includes or excludes columns with certain data types from a DataFrame

    Parameters
    ----------
    df: DataFrame
        Data to be filtered
    include: list of type or None
        List of data types to be included
    exclude: list of type or None
        List of data types to be excluded

    Example
    -------
    filter_df_types(df, include=[int, float], exclude=[object, datetime])

    Returns
    -------
    A DataFrame with equal or less columns, depending on the inclusion and exclusion
    """
    return df.select_dtypes(include=include, exclude=exclude)


def remove_equal_cols(df):
    """Drops all redundant unique valued columns from a DataFrame, checking column after column"""
    for col in df.columns:
        if len(df[col].unique()) <= 1:
            df.drop(col, axis=1, inplace=True)

    return df


def detect_boolean(df):
    """Checks for all columns the ones that have all 1 and/or 0 to convert it into a boolean dtype"""
    for col in df.columns:
        uniques = set(df[df[col].notnull()][col].unique())
        if (len(uniques) > 0) & (len((uniques - {1, 0, "1", "0"})) == 0):
            df[col] = df[col].astype(bool)

    return df


def detect_numerical(df):
    """Attempts to convert each dtype object column into a float or an int"""
    temp_df = filter_df_dtypes(df, include=[object], exclude=None)
    int_col = ""
    for col in temp_df.columns:
        is_int = False
        try:
            float_col = temp_df[col].astype(float)
        except ValueError:
            continue
        try:
            int_col = temp_df[col].astype(int)
            if pd.Series(int_col == float_col).sum() == len(int_col):
                is_int = True
        except ValueError:
            pass
        if is_int:
            df[col] = int_col
        else:
            df[col] = float_col
    return df


def dtype_infer(df, date_col=None):
    """Function that makes a basic cleaning of the data

    Parameters
    ----------
    df: DataFrame
        pandas DataFrame with all the data for this preparation
    date_col: str
        If provided, converts the column to a date

    Returns
    -------
    A pandas DataFrame with all the columns inferred

    Notes
    -----
    This is a suggestion function and may change in time a lot
    """
    df = detect_numerical(df)

    df = detect_boolean(df)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])

    return df
