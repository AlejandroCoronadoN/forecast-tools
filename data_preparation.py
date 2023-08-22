"""Miscellaneous tool functions"""
import logging
from numpy import float64, bool, nan
import pandas as pd
import pdb
import itertools
import collections
from multiprocessing import Pool, cpu_count
from aiqutils.data_cleaning import remove_equal_cols, filter_df_dtypes, dtype_infer

logging.basicConfig(level=logging.INFO)


class AbraxasToolsException(Exception):
    """This is an Exception raise when something goes terribly wrong while executing."""
    pass


def create_date_dummies(df, date_col):
    """Get the most info from your date column. Creates dummies (boolean) columns for:
    hour: from 1 to 24 (up to 24 new cols)
    weekday: from monday to sunday (up to 7 new cols)
    month: month name (up to 12 new cols)
    """

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except TypeError:  # DateIndex need another type of processing
        df[date_col] = df[date_col].apply(lambda x: x.to_timestamp())

        # Hours
    hours_df = pd.get_dummies(df[date_col].dt.hour).astype(bool)
    hours_df = remove_equal_cols(hours_df)
    if len(hours_df.columns) > 0:
        hours_df.columns = date_col + "_daytime_" + hours_df.columns
        df = df.join(hours_df)

    # Weekdays
    weekday_df = pd.get_dummies(df[date_col].dt.strftime("%a")).astype(bool)
    weekday_df = remove_equal_cols(weekday_df)
    if len(weekday_df.columns) > 0:
        weekday_df.columns = date_col + "_" + weekday_df.columns
        df = df.join(weekday_df)

    # Months
    month_df = pd.get_dummies(df[date_col].dt.strftime("%b")).astype(bool)
    month_df = remove_equal_cols(month_df)
    if len(month_df.columns) > 0:
        month_df.columns = date_col + "_" + month_df.columns
        df = df.join(month_df)

    return df


def _make_dummies(df, col):
    """Generate boolean columns for each unique value on the col (dummy)"""
    dummyfied_col = pd.get_dummies(df[col], prefix=col)
    df = df.join(dummyfied_col)

    return df


def create_categorical_dummies(df, cols=None, max_uniques=5):
    """Create dummies from categorical (object) cols only if it has equal or
    less uniques than max_uniques

    Parameters
    ----------
    df: DataFrame
        DataFrame with the data
    cols: list of str
        List of the cols to create dummies
        If none given, then it will take all the columns
    max_uniques: int
        Maximum number of uniques inside the column.
        If it is bigger, then no column will be created
    """
    columns_to_check = df.columns if not cols else cols

    #categorical_cols = filter_df_dtypes(df[columns_to_check], include=[object])
    categorical_cols = columns_to_check
    #print("TEST create_categorical_dummies: '\n\t columns_to_check: {}  \n\tcategorical_cols{} ".format(columns_to_check, categorical_cols))
    for col in categorical_cols:
        if len(df[col].unique()) <= max_uniques:
            df = _make_dummies(df, col)

    return df


def _process_with_freq(arg_list):
    """Process that creates the shift using frequencies.

    Process
    -------
    1. Check if lag == 0. This is due a bug in pandas that doesn't do anything
    2. Create the shift directly using tshift function
    3. Rename the columns to the fitting new names
    4. apply the Roll requested using Exponential Weighted Mean (ewm)

    Returns
    -------
    A DataFrame with the numerical columns shifted in time, aggregated by
    roll param and weighted using ewm
    """

    lag = arg_list[0]
    roll = arg_list[1]
    numerical_cols = arg_list[2]
    group_nodate = arg_list[3]
    new_colnames = arg_list[4]
    rolling_function = arg_list[5]
    freq = arg_list[6]

    if lag == 0:  # Pandas 0.23.3 bug :(
        temp_df = grouped_sum[numerical_cols + group_nodate].tshift(-1, freq=freq)
        temp_df = temp_df.tshift(1, freq=freq)
    else:
        temp_df = grouped_sum[numerical_cols + group_nodate].tshift(lag, freq=freq)
    temp_df = temp_df.rename(columns=dict(zip(numerical_cols, new_colnames)))
    temp_df = temp_df.reset_index()
    grouped_sum_lag = temp_df.groupby(group_nodate)

    # ------------------------------Roll---------------------------
    if rolling_function == "ewm":
        temp_df[new_colnames] = _ewm_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function == "rolling":
        temp_df[new_colnames] = _rolling_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function =="expanding":
        temp_df[new_colnames] = _expanding_function(grouped_sum_lag, new_colnames, roll)
    else:
        raise Exception("'rolling_function {} is not implemented".format(rolling_function))

    return temp_df


def _process_wo_freq(arg_list):
    """Process that creates the shift without using frequencies.

    Process
    -------
    1. Use the grouped sum and shift them by "lag" spaces
    2. Group by again, this time not using the date
    3. Apply Exponential Weighted Average (ewm) using the roll

    Returns
    -------
    A DataFrame with the numerical columns shifted in spaces, aggregated by
    roll param and weighted using ewm and also the name of the new columns

    NOTES
    -----
    A) This function WILL NOT create future data because it would need to
        invent next spaced categorical data
    B) This function WILL NOT create gaps (e.g. [1, 3] will lag 3 to 1 not
        considering 2)
    """

    lag = arg_list[0]
    roll = arg_list[1]
    numerical_cols = arg_list[2]
    group_nodate = arg_list[3]
    new_colnames = arg_list[4]
    rolling_function = arg_list[5]

    # ------------------------------Lag----------------------------
    t_ser[new_colnames] = grouped_sum[numerical_cols].shift(lag)
    grouped_sum_lag = t_ser.groupby(group_nodate)

    # ------------------------------Roll---------------------------
    if rolling_function == "ewm":
        temp_df = _ewm_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function == "rolling":
        temp_df = _rolling_function(grouped_sum_lag, new_colnames, roll)
    elif rolling_function == "expanding":
        temp_df = _expanding_function(grouped_sum_lag, new_colnames, roll)
    else:
        raise Exception("rolling_function {} is not implemented".format(rolling_function))


    return new_colnames, temp_df

def _ewm_function(grouped_sum_lag, new_colnames, roll):
    temp_df = grouped_sum_lag[new_colnames].apply(
        lambda x: x.ewm(span=roll).mean()).reset_index(drop=True)
    return temp_df

def _rolling_function(grouped_sum_lag, new_colnames, roll):
    temp_df = grouped_sum_lag[new_colnames].apply(
        lambda x: x.rolling(roll).mean()).reset_index(drop=True)
    return temp_df

def _expanding_function(grouped_sum_lag, new_colnames, roll):
    temp_df = grouped_sum_lag[new_colnames].apply(
        lambda x: x.expanding(roll).mean()).reset_index(drop=True)
    return temp_df



def fill_timeseries( df, id_columns, date_col, freq = "D", fillmethod = "zeros"):
    """ Function that fill all the ommited observations for freq reported observations. 

    Process
    -------
    1. for each ID the start and end date is calculated. 
    2. A subset of the df is created using just a single ID.
        2.a An agregation over the id_columns + date_col is performed to avoid
            double indexing problems.
    3. A DataFrame is created susing the complete dates between start_date
       and end_date in intervals of freq time.
    4. The dates DataFrame is used a baseline and it's merge with the subset df by date.
    5. If the are some dates taht are

    Parameters
    ----------
    df: DataFrame
        Data to be processed
    id_columns: list
        List of the variables that identifies each row as a independent value.
        Date must be excluded.
    date_col: str
        Column name for the  date column.
    freq: str
        Frequency od the obseravtions.
            'D' - if reported daily.
            'M' - if reported monthly ...
    fillmethod: str:
        zeros :replace all non reported values with 0.
        mean  :replace all non reported values with the average of the reported values.
        ffill :replace all non reported values using the closest reported value.


    Returns
    -------
    A DataFrame with complete number of observations, filling the nan values 
    with the last value or     0.

    Notes:
    -------
    It's possible to rebuild this function using parallel programming.

    """
    df_result = pd.DataFrame()
    df[date_col] =  pd.to_datetime(df[date_col])    
    df["ID"] = ""
    for col in id_columns:
            df["ID"] = df.ID.apply(str) + "_" + col +df[col].apply(str)

    for id_variable in df.ID.unique():
        id_columns_date = id_columns + [date_col]
        df_subset = df[df.ID == id_variable]
        df_subset = df_subset.groupby(id_columns_date).sum()
        df_subset = df_subset.reset_index()

        start_date = df_subset[date_col].min() 
        end_date = df_subset[date_col].max()
        idx=pd.date_range(start=start_date,end=end_date, freq='D')
        dates = pd.DataFrame(idx).rename(columns ={0:"Date"})

        if fillmethod == "zeros":
            id_values = dict()
            for col in id_columns:
                id_values[col] = df_subset[col].unique()[0]

            df_subset = df_subset.set_index(df_subset[date_col],drop=True)
            df_subset = df_subset.reindex(idx)
            df_subset = df_subset.replace(nan, 0)

            for col in id_columns:
                df_subset[col] = id_values[col] 

            df_subset = df_subset.sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})

        elif fillmethod == "mean":
            id_values = dict()
            for col in id_columns:
                id_values[col] = df_subset[col].unique()[0]

            df_subset = df_subset.reindex(idx)
            df_subset.replace(nan, 0)

            for col in id_columns:
                df_subset[col] = id_values[col] 
            df_subset.reindex(idx).fillna(df.mean()).sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})

        elif fillmethod == "ffil":
            df_subset=df_subset.set_index(df_subset[date_col],drop=True)
            df_subset.reindex(idx).fillna(method='ffill').sort_index(ascending=False).drop( date_col,1).reset_index().rename(columns={'index':date_col})
        else:
            raise Exception("'fllmethod {} is not implemented".format(fillmethod))

        if len(df_result) == 0:
            df_result = df_subset
        else:
            df_result = df_result.append(df_subset)
    return df_result

def get_id(df, id_columns):
    """ Function that returns an ID for each observation in the dataframe using the id_columns"""

    id_variable = pd.DataFrame()
    for col in id_columns:
        if len(id_variable) == 0:
            id_variable = df[col].apply(str)
        else:
            id_variable = id_variable.apply(str) + "_" + df[col].apply(str)
    return id_variable

def interact_categorical_numerical(df, lag_col, numerical_cols,
                                   categorical_cols, lag_list,
                                   rolling_list, agg_funct="sum",
                                   rolling_function = "ewm", freq=None,
                                   group_name=None, store_name=False):
    """Function that takes a list of categorical values and generates
    all the rolling averages for each lag given. If using date, frequency
    bins can be provided

    Parameters
    ----------
    df: DataFrame
        Data to be processed
    lag_col: str
        Column name for the lag (usually the date column)
    numerical_cols: list of str
        Column names for the value from where the metrics are calculated
    categorical_cols: list of str
        List containing the names of the categorical cols to be grouped by
    lag_list: list of int
        List containing each lag to be applied (only for date data).
        This lag is applied with a pandas shift
    rolling_list: list of int
        List containing each rolling list to be applied.
        This is the number of values that will calculate the metric.
        If values are less than this number, a NaN is reported
    agg_funct: str
        Current functions supported are only 'sum' and 'count', which are
        the aggregator functions executed over numerical_cols
    rolling_function:
        Rolling window calculations.
            rolling: equal weghted
            ewm: exponential weighted.
            expanding: expanding weight
    freq: str
        If none, no transformation is done during group by
        Frequency capital letters as stated here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        Setting freq to the observations periodicity it's encourage, otherwise 
        if some value it's no reported at a particular date the function will
        bot consider this date for the aggregation process. 
    group_name: str
        base name for the columns creator.
        If none given, it will make a long identifiable name
    store_name: bool
        If True, the resulting cols will only have the original colname
        If False, a structured history-keeping name is given

    Returns
    -------
    The same input df with all the new columns
    """
    global t_ser
    global grouped_sum

    if not group_name:
        group_name = "{cat_cols}".format(
            cat_cols="_".join(categorical_cols))

    group_nodate = list(set(categorical_cols) - {lag_col})
    categorical_cols = group_nodate + [lag_col]

    if freq is not None:
        group_by_arg = group_nodate + [pd.PeriodIndex(df[lag_col], freq=freq)]
    else:   
        group_by_arg = categorical_cols

    grouped = df.groupby(group_by_arg)

    t_ser = grouped[numerical_cols]

    if agg_funct == "sum":
        t_ser = t_ser.sum().sort_index()
    elif agg_funct == "count":
        t_ser = t_ser.count().sort_index()
    else:
        raise Exception("'{0}' is not implemented".format(agg_funct))

    t_ser = t_ser.reset_index()

    if freq:
        t_ser[lag_col] = t_ser[lag_col].apply(lambda x: x.to_timestamp())
        grouped_sum = t_ser.set_index(lag_col)
        t_set = t_ser[categorical_cols].head(0)
    else:
        grouped_sum = t_ser.groupby(group_nodate)
        t_set = t_ser[categorical_cols]

    col_naming = "{col}" if store_name else \
        "{funct}_{group}_on_{col}_with_{process}_{rolling_function}"
    thread_params = []
    for lag in lag_list:
        for roll in rolling_list:

            process = "roll{n_roll}_lag{n_lag}".format(n_roll=roll,
                                                       n_lag=lag)
            new_colnames = [col_naming.format(funct=agg_funct,
                                              group=group_name, col=col,
                                              process=process, rolling_function = rolling_function)
                            for col in numerical_cols]
            logging.info("new_cols: {0}".format(", ".join(new_colnames)))

            if len(new_colnames) > 0:
                thread_params.append((lag, roll, numerical_cols,
                                      group_nodate, new_colnames, rolling_function, freq))

    pool = Pool(cpu_count() - 1)
    if freq:
        temp_dfs = pool.map(_process_with_freq, thread_params)
        for each_df in temp_dfs:
            t_set = t_set.merge(each_df, how="outer", on=categorical_cols)
    else:
        temp_dfs = pool.map(_process_wo_freq, thread_params)
        for new_cols, each_df in temp_dfs:
            t_set[new_cols] = each_df

    if (lag_col in t_set.columns) & (lag_col not in categorical_cols):
        t_set.drop(lag_col, axis=1, inplace=True)
        
    pool.close()
    del t_ser
    del grouped_sum

    return t_set


def _get_null_countable(df, min_nulls):
    """Check if categorical columns have null values and see if it can be
    counted to extract as much information as possible from it"""
    null_countables = []
    for col in filter_df_dtypes(df, include=[object]).columns:
        if (df[col].isnull()).sum() / float(len(df)) > min_nulls:
            null_countables.append(col)
    return null_countables


def _get_summable_cols(df, cols_to_include=None, cols_to_exclude=None):
    """Summable cols are any dtype that can be summed

    Parameters
    ----------
    df: DataFrame
        Data input
    cols_to_include: list of str
        Columns to consider when getting which ones can be summed
        If none given, all columns will be considered
    cols_to_exclude: list of str
        Columns to not even look at in this function
    """
    cols_to_include = df.columns if cols_to_include is None else cols_to_include
    cols_to_exclude = {} if cols_to_exclude is None else set(cols_to_exclude)

    summable_cols = filter_df_dtypes(df[cols_to_include],
                                     include=[bool, int, float,
                                              float64]).columns
    summable_cols = list(set(summable_cols) - set(cols_to_exclude))
    return summable_cols

def polynomial_features(df, numerical_cols, grade=2 ):
  """ This function take a list of numerical columns and calculates all
  the posible combinations polynomial features of grade = grade.

    Parameters
    ----------
    df: DataFrame
        Data to be processed
    numerical_cols: str
        list of the varaibles that will be added or substracted.
    grade: int
        number of varaibles that will be added or substracted 
        to create the output features.
    operation: str
        add, substract
    
    Output:
    A DataFrame with the new polynomial variables produced 
    by the multiplication of the numerical_cols.
  """
  polynomial_list = []
  for col in numerical_cols:
    for i in range(grade):
      polynomial_list.append(col)

  polynomial_variables = list(itertools.combinations(polynomial_list, grade))
  polynomial_variables = set(polynomial_variables)
  for variable in polynomial_variables:
    variable_dict = dict(collections.Counter(variable))
    variable_name = ""
    print(variable_dict)
    for key in variable_dict.keys():
      power = variable_dict[key]
      if variable_name == "":
        variable_name = str(key) + "power" +  str(power)
      else:
        variable_name = variable_name + "_" + str(key) + "power" + str(power)
    df[variable_name] = 1
    for feat in variable:
      df[variable_name] = df[variable_name] * df[feat]
  return df



def add_substract_features(df, numerical_cols, grade=2, operation = "add" ):
  """ This function take a list of numerical columns and calculates all
  the posible combinations of grade features in the numerical_cols list
  and performs and operation.

    Parameters
    ----------
    df: DataFrame
        Data to be processed
    numerical_cols: str
        list of the varaibles that will be added or substracted.
    grade: int
        number of varaibles that will be added or substracted 
        to create the output features.
    operation: str
        add, substract
    
    Outpu:
    ----------
    A DataFrame with the new variables produced by the addition or
    substraction of the numerical_cols        
  """
  addsubstract_variables  =list()
  for i in range(grade+1):
    if i <2:
      continue
    else:
      addsubstract_variables.extend(list(itertools.combinations(numerical_cols, i)))
  addsubstract_variables = set(addsubstract_variables)
  for variable in addsubstract_variables:
    name = ""

    if operation == "add":
      for element in variable:
        if name == "":
          name = element
        else:
          name = name + "_plus_" + element
      df[name] = 0
      print(name)
      for element in variable:
          df[name] = df[name] + df[element]

    elif operation == "substract":
      for element in variable:
        if name == "":
          name = element
        else:
          name = name + "_minus_" + element
      df[name] = 0
      print(name)
      for element in variable:
          df[name] = df[name] - df[element] 
    else:
      raise Exception("Invalid operation {}".format(operation))
  return df


def smart_timelag_groupby(df, target_col, groupby_cols, min_nulls, lag_col,
                          lag, freq=None):
    """Aggregation function that in the end returns data aggregated by
    the groupby_cols.

    Process
    -------
    1.- get which columns can be summed and which ones can be counted
    2.- create a new DataFrame
    3.- add the original target col to df without lag with a group by sum
    4.- add the cols from 1.- lagged by lag and grouped

    Parameters
    ----------
    df: DataFrame
        data to be used
    target_col: str
        name of the column that will not receive lag
    groupby_cols: list of str
        list of the columns that will be used for pandas groupby
    min_nulls: float
        minimum fraction of nulls to be considered a countable column
    lag_col:
        name of the column to be used as a base of lag.
        Can be date or any sortable column
    lag: int
        Lag used for all non-target variables (as this is meant to be a
        forecast)
    freq: str
        If none, no transformation is done during group by
        Frequency capital letters as stated here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Returns
    -------
    a new df aggregated by groupby_cols and date, but lagged by lag in all
    but the target column.

    Notes
    -----
    Any categorical (object) col that does not pass the min_nulls threshold
    will inevitably be ignored due to the grouping nature of this process
    """
    null_countables = _get_null_countable(df, min_nulls=min_nulls)
    summable_cols = _get_summable_cols(df, cols_to_exclude=groupby_cols + [lag_col])
    original_df = df

    groupby_cols = list(set(groupby_cols).union({lag_col}))

    df = interact_categorical_numerical(df=original_df, lag_col=lag_col,
                                        numerical_cols=[target_col],
                                        categorical_cols=groupby_cols,
                                        lag_list=[0], rolling_list=[1],
                                        agg_funct="sum", store_name=True,
                                        freq=freq)

    temp_df = interact_categorical_numerical(df=original_df, lag_col=lag_col,
                                             numerical_cols=null_countables,
                                             categorical_cols=groupby_cols,
                                             lag_list=[lag], rolling_list=[1],
                                             agg_funct="count", store_name=True,
                                             freq=freq)

    df = pd.merge(df, temp_df, how='outer', on=groupby_cols,
                  suffixes=["", "_lagged"])

    temp_df = interact_categorical_numerical(df=original_df, lag_col=lag_col,
                                             numerical_cols=summable_cols,
                                             categorical_cols=groupby_cols,
                                             lag_list=[lag], rolling_list=[1],
                                             agg_funct="sum", store_name=True,
                                             freq=freq)

    df = pd.merge(df, temp_df, how='outer', on=groupby_cols,
                  suffixes=["", "_lagged"])

    return df


def split_lagged_df(df, date_col, target_col, min_lag, freq):
    train_df = df[df[target_col].notnull()]
    max_data_date = train_df[date_col].max()
    periods = pd.date_range(max_data_date, periods=min_lag, freq=freq)
    horizon = pd.to_datetime(periods[-1])
    future_df = df[df[target_col].isnull() &
                   (df[date_col] >= horizon)]

    return train_df, future_df



def prepare_lag_data(df, target_col, groupby_cols, lag_col,
                     lag_list, rolling_list, freq=None, min_nulls=0.5,
                     return_split=False):
    """Suggested data preparation for production algorithms using naive lag

    Parameters
    ----------
    df: DataFrame
        data to be used
    target_col: str
        name of the column that will not receive lag
    groupby_cols: list of str
        list of the columns that will be used for pandas groupby
    freq: str
        If none, no transformation is done during group by
        Frequency capital letters as stated here:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    lag_col: str
        name of the column to be used as a base of lag.
        Can be date or any sortable column
    lag_list: list of int
        list containing all different lags user wants to use in this process
    rolling_list: list of int
        list containing all different rolls (how many values you wish to use
        when calculating backwards)
    min_nulls: float
        minimum fraction of nulls to be considered a countable column
    return_split: bool
        if True, returns two DataFrames instead of one. The first one is the
        train DataFrame and the other one is for prediction use.
        NOTE: this argument only works if freq is active too!
    Returns
    -------
    A DataFrame containing many new columns and all of them lagged
    """

    df = dtype_infer(df)

    df = create_categorical_dummies(df, max_uniques=5)

    df = smart_timelag_groupby(df,
                               target_col=target_col,
                               groupby_cols=groupby_cols,
                               min_nulls=min_nulls,
                               lag_col=lag_col,
                               freq=freq,
                               lag=min(lag_list))

    if len(lag_list) > 1:
        df.drop(target_col + "_lagged", axis=1)
        temp_df = interact_categorical_numerical(df, lag_col=lag_col,
                                                 numerical_cols=[target_col],
                                                 categorical_cols=groupby_cols,
                                                 lag_list=lag_list,
                                                 freq=freq,
                                                 rolling_list=rolling_list)
        df = df.merge(temp_df, how="outer", on=groupby_cols + [lag_col])

    df = create_date_dummies(df, date_col=lag_col)

    df = remove_equal_cols(df)

    if (not return_split) & (freq is None):
        return df
    else:
        train, future = split_lagged_df(df, date_col=lag_col,
                                        target_col=target_col,
                                        min_lag=min(lag_list),
                                        freq=freq)
        return train, future