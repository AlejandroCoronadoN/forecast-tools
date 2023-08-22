

import pandas as pd 
import numpy as np
import copy
from aiqutils import price_models 
from aiqutils import data_preparation
import time 
import pdb
import pickle

def  transform_KFold(df,  y_column, K):
    """
    Divides a DataFrame into K sets using the index. 
    df_kfolded[fold_number]
    
        df_kfolded[fold_number]["index"]: randomly selected index 
                                          for fold_number fold.

        df_kfolded[fold_number]["y"]:     subset of the target variable indexed
                                          by the selected index.

        df_kfolded[fold_number]["data"]:  subset of the DataFrame indexed
                                          by the selected index. 
                                          target variable excluded

        **df_kfolded["all_data"][...]: group all the infromation.

        **df_kfolded["train_test"][...]: firt k-1 fold are grouped into a single
            DataFrame for tarining and las K fold is used as test set. 
    """
    df = df.reset_index( drop = True)
    all_index =  df.index
    df_kfolded = dict()
    obs_per_fold = int(np.round(len(df)/K))

    for k in range(K):
        print("K: {}  all_index: {} ".format(k, len(all_index)))
        df_kfolded[k] = dict()
        df_copy = df.copy()
        if k == K-1:
            df_kfolded[k]["y"] = df_copy.loc[all_index, y_column]
            del df_copy[y_column]
            df_kfolded[k]["data"]  = df_copy.loc[all_index]
            df_kfolded[k]["index"] = all_index
            if y_column in df_kfolded[k]["data"]:
                print("\n\n\nERROR: Target variable in df_kfolded[k][data]")
        else:

            fold_index = np.random.choice(all_index, obs_per_fold, replace= False)
            fold_data  = df_copy.loc[fold_index]
            fold_y     = fold_data[y_column]
            del fold_data[y_column]

            all_index = [item for item in all_index if item not in fold_index]

            df_kfolded[k]["index"] = fold_index
            df_kfolded[k]["data"]  = fold_data
            df_kfolded[k]["y"]     = fold_y

            if y_column in df_kfolded[k]["data"]:
                print("\n\n\nERROR: Target variable in df_kfolded[k][data]")

        print( len(df_kfolded[k]["index"]))

    df_copy = df.copy()
    df_kfolded["all_data"] = dict()
    df_kfolded["all_data"]["index"] =  all_index
    df_kfolded["all_data"]["y"]     =  df_copy[y_column]
    del df_copy[y_column]
    df_kfolded["all_data"]["data"]  =  df_copy

    df_kfolded["train"] = dict()
    df_kfolded["test"]  = dict()
    df_kfolded["train"]["data"] = pd.DataFrame() 

    for k in range(K):
        if k != K-1:
            if len(df_kfolded["train"]["data"]) > 0:
                df_kfolded["train"]["data"] = df_kfolded["train"]["data"].append(df_kfolded[k]["data"] ) 
                df_kfolded["train"]["y"] = df_kfolded["train"]["y"].append(df_kfolded[k]["y"] ) 
                df_kfolded["train"]["index"] = df_kfolded["train"]["y"].append( pd.DataFrame(df_kfolded[k]["index"]))

            else:
                df_kfolded["train"]["data"] = df_kfolded[k]["data"]
                df_kfolded["train"]["y"] = df_kfolded[k]["y"]
                df_kfolded["train"]["index"] = pd.DataFrame(df_kfolded[k]["index"])

        else:
            df_kfolded["test"]["data"] = df_kfolded[k]["data"]
            df_kfolded["test"]["y"] = df_kfolded[k]["y"]
            df_kfolded["test"]["index"] = pd.DataFrame(df_kfolded[k]["index"])

    return df_kfolded


def df_partition(df, id_columns, dummies_cols, agregation_list, date_col, delete_aggregation_list =True ):
    """Create a dictionary with an entry for each group in the agregation_list.
    This dictionaary is used in the price optimization process. In each entry of
    the dicionary there is a dataframe with the subset of information for each 
    group in agregation_list.

    Parameters
    ----------
    df: DataFrame
    agregation_list: level of aggregation for the price optimization and price
    elasticity analysis.
    delete_aggregation_list: If True it excludes the columns of agregation_list from
    PRICE_ELASTICITY_ANALYSIS[forall]["data"]

    Returns
    -------
    Dictioanry with the following structure:
        If aggregation = ["group_A", "group_B"] then each entry will have a subset
        PRICE_ELASTICITY_ANALYSIS["element_in_group_A_element_in_group_B"] = { 
            ["data"] = df[(df.group_A == element_in_group_A) & (df.group_B == element_in_group_B)]

    """
    agregation_list_copy = agregation_list.copy()
    PRICE_ELASTICITY_ANALYSIS = dict()      
    df_subpartition(df, id_columns, dummies_cols, date_col, "", agregation_list_copy, PRICE_ELASTICITY_ANALYSIS)
    return PRICE_ELASTICITY_ANALYSIS


def _delete_aggregation_list_from_PRICE_ANALYSIS( PRICE_ELASTICITY_ANALYSIS , agregation_list):
    """excludes the columns of agregation_list from
    PRICE_ELASTICITY_ANALYSIS[forall]["data"]
    """
    for group in PRICE_ELASTICITY_ANALYSIS.keys():
        df_without_agg = PRICE_ELASTICITY_ANALYSIS[group]["data"].drop(agregation_list, axis =1)
        PRICE_ELASTICITY_ANALYSIS[group]["data"] = df_without_agg

    return PRICE_ELASTICITY_ANALYSIS

def df_subpartition(df_subset, id_columns, dummies_cols, date_col, name, agregation_list, PRICE_ELASTICITY_ANALYSIS):
    """Recursive function used by df_subpartition
    """
    if len(agregation_list) == 0:
        PRICE_ELASTICITY_ANALYSIS[name] = dict()
        PRICE_ELASTICITY_ANALYSIS[name]["date"] = df_subset[date_col]
        PRICE_ELASTICITY_ANALYSIS[name]["data_id"] = df_subset[id_columns]
        PRICE_ELASTICITY_ANALYSIS[name]["dummies_cols"] = df_subset[dummies_cols]
        df_subset = df_subset.drop(id_columns, axis =1)
        dummies_to_delete =  [x for x in dummies_cols if x not in id_columns]
        df_subset = df_subset.drop(dummies_to_delete, axis =1)

        PRICE_ELASTICITY_ANALYSIS[name]["data"] = df_subset

        return PRICE_ELASTICITY_ANALYSIS

    else:
        agg = agregation_list[0]
        del agregation_list[0]
        for group in df_subset[agg].unique():
            df_subset_next_iteration =  copy.deepcopy(df_subset)
            subgroup_name = name+"_" + str(agg) + str(group)
            df_subset_next_iteration = df_subset_next_iteration[df_subset_next_iteration[agg] == group]
            #print("TEST group: {} df_subset: {} ".format(group, df_subset_next_iteration.shape))
            PRICE_ELASTICITY_ANALYSIS = df_subpartition(
                                                df_subset_next_iteration, 
                                                id_columns, 
                                                dummies_cols, 
                                                date_col,
                                                subgroup_name, 
                                                agregation_list, 
                                                PRICE_ELASTICITY_ANALYSIS)
        return PRICE_ELASTICITY_ANALYSIS


def fit_price_model(PRICE_ELASTICITY_ANALYSIS, model, demand_col,  all_data = False):
    """Creates a new element in PRICE_ELASTICITY_ANALYSIS["element_in_group_A_element_in_group_B"] 
        ["model_name"] = fitted model

    """
    for group in PRICE_ELASTICITY_ANALYSIS.keys():
        df =  PRICE_ELASTICITY_ANALYSIS[group]["data"]
        df_kfolded =  transform_KFold(df, demand_col, 5)

        if all_data:
                X_train = df_kfolded["all_data"]["data"]
                y_train = df_kfolded["all_data"]["y"]
                X_test = df_kfolded["test"]["data"]
                y_test = df_kfolded["test"]["y"]

        else: #TO DO:  esta parte esta incompleta y no hace nada de sentido
                X_train = df_kfolded["train"]["data"]
                y_train = df_kfolded["train"]["y"]
                X_test = df_kfolded["test"]["data"]
                y_test =  df_kfolded["test"]["y"]

        #pbd.set_trace()
        model.fit(X_train, y_train, X_test, y_test, error_type = "MAE")
        model_name = model.name
        PRICE_ELASTICITY_ANALYSIS[group][model_name] = dict()
        PRICE_ELASTICITY_ANALYSIS[group][model_name]["fitted_model"] = model
        PRICE_ELASTICITY_ANALYSIS[group][model_name]["fitted_columns"] = X_train.columns

        print("MAE: ", np.mean(np.abs(y_test - model.predict(X_test))))
            
def price_analysis(PRICE_ELASTICITY_ANALYSIS, model_name, price_col, demand_col, date_col, analysis_type,
    price_increments_function, interact_categorical_numerical_parameters, percentual_range=10,
    evaluation_perido_start = None, evaluation_perido_end= None):
    """Creates  a new element in PRICE_ELASTICITY_ANALYSIS["element_in_group_A_element_in_group_B"]["model_name"]
        ["elasticities_analysis"] =
            [mean_price + %change] 
            [mean_price]
            [mean_price - %change]

        ["priceoptimization_by_pricevariation"] =
            [mean_price + %change]["demans"]
                                  ["sales"]
            [mean_price]          ["demand"]
                                  ["sales"]
            [mean_price - %change]["demand"]
                                  ["sales"]

        Given an interval of time it measure the impact of a small change in price_col over the objective variable
        demand_col. 

        #TO DO: Build elasticities_by_percentil
    """
    print("analysis_type: ", analysis_type)
    for group in PRICE_ELASTICITY_ANALYSIS.keys():
        print(group)
        fitted_model = PRICE_ELASTICITY_ANALYSIS[group][model_name]["fitted_model"]        
        df_evaluation =  PRICE_ELASTICITY_ANALYSIS[group]["data"].copy()
        df_evaluation = df_evaluation.merge( PRICE_ELASTICITY_ANALYSIS[group]["data_id"], right_index = True, left_index= True)

        if evaluation_perido_start:
            df_evaluation = df_evaluation[df["DATE"] >= evaluation_perido_start]
        if evaluation_perido_end:
            df_evaluation = df_evaluation[df["DATE"] <= evaluation_perido_end]

        del df_evaluation[demand_col]

        if analysis_type == "elasticities_by_pricevariation":
            elasticities_by_pricevariation(PRICE_ELASTICITY_ANALYSIS, 
                group =group, 
                df = df_evaluation, 
                price_col= price_col, 
                model = fitted_model, 
                percentual_range = percentual_range,
                price_increments_function = price_increments_function,
                interact_categorical_numerical_parameters = interact_categorical_numerical_parameters
                )
        elif analysis_type == "priceoptimization_by_pricevariation": 
            priceoptimization_by_pricevariation(PRICE_ELASTICITY_ANALYSIS, 
                group =group, 
                df = df_evaluation, 
                price_col= price_col, 
                model = fitted_model, 
                percentual_range = percentual_range,
                price_increments_function = price_increments_function,
                interact_categorical_numerical_parameters = interact_categorical_numerical_parameters
                )
        else:
            print("ERROR: analysis_type not defined ")




def increase_prices_with_categorical_numerical(df, interact_categorical_numerical_parameters, price_col, pct):
    """ Increased all the lagged and power variables applied in the data_preparation 
    interact_categorical_numerical output
    """
    df = df.drop(interact_categorical_numerical_parameters["selected_columns"] , axis =1 )
    df[price_col] = (1+pct)*df[price_col]
    id_columns    = interact_categorical_numerical_parameters["id_columns"]
    for roll_fun in interact_categorical_numerical_parameters["rolling_functions"]:
        df_rolling_function = data_preparation.interact_categorical_numerical(
                                           df,
                                           interact_categorical_numerical_parameters["lag_col"],
                                           interact_categorical_numerical_parameters["numerical_cols"],
                                           interact_categorical_numerical_parameters["categorical_cols"],
                                           interact_categorical_numerical_parameters["lag_list"] ,
                                           interact_categorical_numerical_parameters["rolling_list"],
                                           interact_categorical_numerical_parameters["agg_funct"],
                                           roll_fun, 
                                           interact_categorical_numerical_parameters["freq"],
                                           interact_categorical_numerical_parameters["group_name"],
                                           interact_categorical_numerical_parameters["store_name"]
                                           )

        selected_columns = [x for x in interact_categorical_numerical_parameters["selected_columns"] if x in df_rolling_function.columns]
        df= df.merge(df_rolling_function[selected_columns + id_columns], on = id_columns )
    return df

def elasticities_by_pricevariation(PRICE_ELASTICITY_ANALYSIS, group, df, price_col, model, percentual_range,
    price_increments_function, interact_categorical_numerical_parameters):
    """This fucntion creates a new dictioanry in  PRICE_ELASTICITY_ANALYSIS[group][model.name]["elasticities_analysis"]
        [pct_index_name]:
            [mean_price + %change] 
            [mean_price]
            [mean_price - %change]    
    This new entry represents a new DataFrame in which each entry reports the elasticity of a given group 
    using model = model arround the mean price plus a percentual increment in price.

    Returns
    ----------
    Dictionary inside PRICE_ELASTICITY_ANALYSIS. Each key of this new dictionary is label with the percentual increment
    made arround the mean price and links it to the respective elasticity.

    """
    mean_price =  df[price_col].mean()
    epsilon = .1 #TODO: Maybe epsilon should be a pased value, the increase the sensibiity in price
    initial_demand = model.predict(df)
    PRICE_ELASTICITY_ANALYSIS[group][model.name]["elasticities_by_pricevariation"] = dict()

    for pct in range(-percentual_range, percentual_range):
        pct = pct/100

        pct_index_name = "price_variation_" + str(pct)
        df_test = df.copy()
        df_test_plusepsilon = df.copy()

        #TO DO: defne a new fucntion to increase prices in the precense of other type of price variables
        #such as price_srwt, price_cubic, ...
        df_test[price_col] = df[price_col] + pct*df[price_col]
        df_test_plusepsilon[price_col] = df[price_col] + (epsilon+pct)*df[price_col] 

        demand_at_pct = model.predict(df_test)
        demand_at_pct_plusepsilon = model.predict(df_test_plusepsilon)

        elasticity = np.mean((demand_at_pct_plusepsilon - demand_at_pct)/demand_at_pct)
        PRICE_ELASTICITY_ANALYSIS[group][model.name]["elasticities_by_pricevariation"][pct_index_name] = elasticity

        #print("\n\nTEST at {} : \n\tpct_price: {} \n\t pct_epsilon".format(pct, df_test[price_col].mean(), df_test_plusepsilon[price_col].mean()))


def priceoptimization_by_pricevariation(PRICE_ELASTICITY_ANALYSIS, group, df, price_col, model, percentual_range, 
    price_increments_function, interact_categorical_numerical_parameters):
    """This fucntion creates a new dictioanry in  PRICE_ELASTICITY_ANALYSIS[group][model.name]["price_optimization"]
        [pct_index_name]:
            [mean_price + %change] 
            [mean_price]
            [mean_price - %change]    
    This new entry represents a new DataFrame in which each entry reports the elasticity of a given group 
    using model = model arround the mean price plus a percentual increment in price.

    Returns
    ----------
    Dictionary inside PRICE_ELASTICITY_ANALYSIS. Each key of this new dictionary is label with the percentual increment
    made arround the mean price and links it to the respective elasticity.

    """
    PRICE_ELASTICITY_ANALYSIS[group][model.name]["priceoptimization_by_pricevariation"] = dict()

    for pct in range(-percentual_range, percentual_range):
        pct = pct/100
        pct_index_name = "price_variation_" + str(pct)
        df_test = df.copy()

        if price_increments_function == "simple_price_increase":
            df_test[price_col] = (1+pct)*df[price_col]
        elif price_increments_function == "increase_prices_with_categorical_numerical": 
            #pdb.set_trace()
            df_test_copy = df_test.copy()
            df_test = increase_prices_with_categorical_numerical(df_test, interact_categorical_numerical_parameters, 
                price_col, pct)
            #pdb.set_trace()

        else:
            print("ERROR the selected price_increments_function is not valid")
        df_test = df_test[PRICE_ELASTICITY_ANALYSIS[group][model.name]["fitted_columns"]]        
        demand_at_pct = np.sum(model.predict(df_test))
        sales_at_pct =  np.sum(demand_at_pct * df_test[price_col])

        PRICE_ELASTICITY_ANALYSIS[group][model.name]["priceoptimization_by_pricevariation"][pct_index_name] = dict()
        PRICE_ELASTICITY_ANALYSIS[group][model.name]["priceoptimization_by_pricevariation"][pct_index_name]["price"] = df_test[price_col].mean()
        PRICE_ELASTICITY_ANALYSIS[group][model.name]["priceoptimization_by_pricevariation"][pct_index_name]["demand"] = demand_at_pct
        PRICE_ELASTICITY_ANALYSIS[group][model.name]["priceoptimization_by_pricevariation"][pct_index_name]["sales"] = sales_at_pct

group = '_MARKET_MEANINGagencia_4e_Sobeys_Maritimes'


def print_price_analysis(PRICE_ELASTICITY_ANALYSIS, analysis_type):
    for group in PRICE_ELASTICITY_ANALYSIS.keys():
        for model_name in PRICE_ELASTICITY_ANALYSIS[group].keys():
            if model_name not in ["date", "data", "data_id", "dummies_cols"]:
                print("\n\n {} MODEL: {}".format(analysis_type, model_name))
                for elast in PRICE_ELASTICITY_ANALYSIS[group][model_name][analysis_type].keys():
                    print(PRICE_ELASTICITY_ANALYSIS[group][model_name][analysis_type][elast]) 
