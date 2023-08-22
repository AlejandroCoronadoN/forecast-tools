
import numpy as np
import pandas as pd 
from pprint import pprint


def calculate_next_chunk(parameters_model, next_chunk):
    print("\n\n\t-------------Generating next hyperparameter:----------------")    
    pprint(parameters_model)
    if next_chunk > parameters_model["chromosome_position"]["end"]:
        return next_chunk
    else:
        return parameters_model["chromosome_position"]["end"]

def print_hyperparameter(parameters):
    for key in parameters.keys():
        print("\n\n\t--------------Parameter: {} ----------------".format(key))
        pprint(parameters[key])

def get_param_values(precision, min_value,max_value):
    """ Returns a set of values that can be used in a particular parameter that lives between
    min_value and max_value.
    """
    if max_value:
        num_values = np.power(2, precision)
        steps = max_value/ num_values
        values = np.arange(min_value, max_value, steps)
        values[len(values)-1] = max_value

    else:
        num_values = np.power(2, precision)
        max_value = min_value + num_values
        steps = max_value/ num_values
        values = np.arange(min_value, max_value, num_values)
        values[len(values)-1] = max_value

    return values


def get_start_position( parameters_model , next_chunk):
    if parameters_model["isDefault"] == True:
        return 0
    else:
        return next_chunk


def get_end_position( parameters_model , next_chunk):
    if parameters_model["isDefault"] == True:
        return 0
    else:
        return next_chunk + parameters_model["bits"]


def _get_xgboost_hyperparams(precision)
    """
    Creates a dictionary with all the hyperparamters of XGBoost model. 
    This dictionary is used in the
    """

    next_chunk = 0
    parameters = dict()
    parameters["booster"] = dict()
    parameters["booster"]["isDefault"] = False
    parameters["booster"]["bits"] = 1
    parameters["booster"]["description"] = \
        "booster [default=gbtree] Select the type of model to run at each iteration. It has 2 options:\
        gbtree: tree-based models gblinear: linear models <----- This kind of booster is never used."
    parameters["booster"]["chromosome_position"] = dict()
    parameters["booster"]["chromosome_position"]["start"] = get_start_position( parameters["booster"] , next_chunk)
    parameters["booster"]["chromosome_position"]["end"] = get_end_position( parameters["booster"] , next_chunk)
    parameters["booster"]["values"] = ["gbtree", "gblinear"]
    next_chunk = calculate_next_chunk( parameters["booster"], next_chunk )

    #Aqui debería ir algo asi como parameters.key : parameters["booster"]["values"][selected_bit]
    parameters["silent"] = dict()
    parameters["silent"]["isDefault"] = True    
    parameters["silent"]["bits"] = 0
    parameters["silent"]["description"] = \
        "Silent mode is activated is set to 1, i.e. no running messages will be printed.\
        It’s generally good to keep it 0 as the messages might help in understanding the model."
    parameters["silent"]["chromosome_position"] = dict()
    parameters["silent"]["chromosome_position"]["start"] = get_start_position( parameters["silent"] , next_chunk)
    parameters["silent"]["chromosome_position"]["end"] = get_end_position( parameters["silent"] , next_chunk)
    parameters["silent"]["values"] = 0
    parameters["silent"]["chromosome_position"]["end"]
    next_chunk = calculate_next_chunk( parameters["silent"], next_chunk )


    parameters["nthread"] = dict()
    parameters["nthread"]["isDefault"] = True
    parameters["nthread"]["bits"] = 0
    parameters["nthread"]["description"] = \
        "This is used for parallel processing and number of cores in the system should be entered."
    parameters["nthread"]["chromosome_position"] = dict()
    parameters["nthread"]["chromosome_position"]["start"] = get_start_position( parameters["nthread"] , next_chunk)
    parameters["nthread"]["chromosome_position"]["end"] = get_end_position( parameters["nthread"] , next_chunk)
    parameters["nthread"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["nthread"], next_chunk )


    parameters["eta"] = dict()
    parameters["eta"]["isDefault"] = False
    parameters["eta"]["bits"] = precision
    parameters["eta"]["description"] = \
        "Analogous to learning rate in GBM. Makes the model more robust by shrinking the weights on each step\
        Typical final values to be used: 0.01-0.2"
    parameters["eta"]["chromosome_position"] = dict()
    parameters["eta"]["chromosome_position"]["start"] = get_start_position( parameters["eta"] , next_chunk)
    parameters["eta"]["chromosome_position"]["end"] = get_end_position( parameters["eta"] , next_chunk)
    parameters["eta"]["values"] = get_param_values(precision, .01, .7)
    next_chunk = calculate_next_chunk( parameters["eta"], next_chunk )



    parameters["min_child_weight"] = dict()
    parameters["min_child_weight"]["isDefault"] = False
    parameters["min_child_weight"]["bits"] = precision
    parameters["min_child_weight"]["description"] = \
        "Defines the minimum sum of weights of all observations required in a child. \
        Higher values - highly specific to the particular sample selected for a tree.\
        Too high values can lead to under-fitting"
    parameters["min_child_weight"]["chromosome_position"] = dict()
    parameters["min_child_weight"]["chromosome_position"]["start"] = get_start_position( parameters["min_child_weight"], next_chunk)
    parameters["min_child_weight"]["chromosome_position"]["end"] = get_end_position( parameters["min_child_weight"], next_chunk )
    parameters["min_child_weight"]["values"] = get_param_values(precision, 1, False)
    next_chunk = calculate_next_chunk( parameters["min_child_weight"], next_chunk )


    parameters["max_depth"] = dict()
    parameters["max_depth"]["isDefault"] = False
    parameters["max_depth"]["bits"] = precision
    parameters["max_depth"]["description"] = \
        "The maximum depth of a tree, same as GBM.\
        Used to control over-fitting as higher depth will allow model to learn relations very specific \
        to a particular sample. Should be tuned using CV. Typical values: 3-10"
    parameters["max_depth"]["chromosome_position"] = dict()
    parameters["max_depth"]["chromosome_position"]["start"] = get_start_position( parameters["max_depth"], next_chunk)
    parameters["max_depth"]["chromosome_position"]["end"] = get_end_position( parameters["max_depth"], next_chunk )
    parameters["max_depth"]["values"] = get_param_values(precision, 1, False)
    next_chunk = calculate_next_chunk( parameters["max_depth"], next_chunk )


    parameters["max_leaf_nodes"] = dict()
    parameters["max_leaf_nodes"]["isDefault"] = False
    parameters["max_leaf_nodes"]["bits"] = precision
    parameters["max_leaf_nodes"]["description"] = \
        "The maximum depth of a tree, same as GBM.\
        Used to control over-fitting as higher depth will allow model to learn relations very specific \
        to a particular sample. Should be tuned using CV. Typical values: 3-10"
    parameters["max_leaf_nodes"]["chromosome_position"] = dict()
    parameters["max_leaf_nodes"]["chromosome_position"]["start"] = get_start_position( parameters["max_leaf_nodes"], next_chunk)
    parameters["max_leaf_nodes"]["chromosome_position"]["end"] = get_end_position( parameters["max_leaf_nodes"], next_chunk )
    parameters["max_leaf_nodes"]["values"] = get_param_values(precision, 1, False)
    next_chunk = calculate_next_chunk( parameters["max_leaf_nodes"], next_chunk )


    parameters["max_leaf_nodes"] = dict()
    parameters["max_leaf_nodes"]["isDefault"] = False
    parameters["max_leaf_nodes"]["bits"] = precision
    parameters["max_leaf_nodes"]["description"] = \
        "The maximum number of terminal nodes or leaves in a tree.\
        Can be defined in place of max_depth. Since binary trees are created, \
        a depth of ‘n’ would produce a maximum of 2^n leaves.\
        If this is defined, GBM will ignore max_depth.0"
    parameters["max_leaf_nodes"]["chromosome_position"] = dict()
    parameters["max_leaf_nodes"]["chromosome_position"]["start"] = get_start_position( parameters["max_leaf_nodes"], next_chunk)
    parameters["max_leaf_nodes"]["chromosome_position"]["end"] = get_end_position( parameters["max_leaf_nodes"], next_chunk )
    parameters["max_leaf_nodes"]["values"] = get_param_values(np.power(2, precision), 1, False) #by definition
    next_chunk = calculate_next_chunk( parameters["max_leaf_nodes"], next_chunk )


    parameters["gamma"] = dict()
    parameters["gamma"]["isDefault"] = True
    parameters["gamma"]["bits"] = 0
    parameters["gamma"]["description"] = \
        "A node is split only when the resulting split gives a positive \
        reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.\
        Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned."
    parameters["gamma"]["chromosome_position"] = dict()
    parameters["gamma"]["chromosome_position"]["start"] = get_start_position( parameters["gamma"], next_chunk)
    parameters["gamma"]["chromosome_position"]["end"] = get_end_position( parameters["gamma"], next_chunk )
    parameters["gamma"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["gamma"], next_chunk )


    parameters["max_delta_step"] = dict()
    parameters["max_delta_step"]["isDefault"] = False
    parameters["max_delta_step"]["bits"] = precision
    parameters["max_delta_step"]["description"] = \
        "In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, \
        it means there is no constraint. If it is set to a positive value, it can help making the \
        update step more conservative. Usually this parameter is not needed, but it might help in logistic \
        regression when class is extremely imbalanced. \
        This is generally not used but you can explore further if you wish."
    parameters["max_delta_step"]["chromosome_position"] = dict()
    parameters["max_delta_step"]["chromosome_position"]["start"] = get_start_position( parameters["max_delta_step"], next_chunk)
    parameters["max_delta_step"]["chromosome_position"]["end"] = get_end_position( parameters["max_delta_step"], next_chunk )
    parameters["max_delta_step"]["values"] = get_param_values(precision,  5, False) #by definition
    next_chunk = calculate_next_chunk( parameters["max_delta_step"], next_chunk )


    parameters["subsample"] = dict()
    parameters["subsample"]["isDefault"] = False
    parameters["subsample"]["bits"] = precision
    parameters["subsample"]["description"] = \
        "Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.\
        Lower values make the algorithm more conservative and prevents overfitting \
        but too small values might lead to under-fitting. Typical values: 0.5-1"
    parameters["subsample"]["chromosome_position"] = dict()
    parameters["subsample"]["chromosome_position"]["start"] = get_start_position( parameters["subsample"], next_chunk)
    parameters["subsample"]["chromosome_position"]["end"] = get_end_position( parameters["subsample"], next_chunk )
    parameters["subsample"]["values"] = get_param_values(precision,  .1, 1) 
    next_chunk = calculate_next_chunk( parameters["subsample"], next_chunk )


    parameters["colsample_bytree"] = dict()
    parameters["colsample_bytree"]["isDefault"] = False
    parameters["colsample_bytree"]["bits"] = precision
    parameters["colsample_bytree"]["description"] = \
        "Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.\
        Typical values: 0.5-1"
    parameters["colsample_bytree"]["chromosome_position"] = dict()
    parameters["colsample_bytree"]["chromosome_position"]["start"] = get_start_position( parameters["colsample_bytree"], next_chunk)
    parameters["colsample_bytree"]["chromosome_position"]["end"] = get_end_position( parameters["colsample_bytree"], next_chunk )
    parameters["colsample_bytree"]["values"] = get_param_values(precision,  .1, 1) 
    next_chunk = calculate_next_chunk( parameters["colsample_bytree"], next_chunk )


    parameters["colsample_bylevel"] = dict()
    parameters["colsample_bylevel"]["isDefault"] = False
    parameters["colsample_bylevel"]["bits"] = precision
    parameters[ "colsample_bylevel"]["description"] = \
        "Denotes the subsample ratio of columns for each split, in each level.\
        I don’t use this often because subsample and colsample_bytree will do the \
        job for you. but you can explore further if you feel so."
    parameters["colsample_bylevel"]["chromosome_position"] = dict()
    parameters["colsample_bylevel"]["chromosome_position"]["start"] = get_start_position( parameters["colsample_bylevel"], next_chunk)
    parameters["colsample_bylevel"]["chromosome_position"]["end"] = get_end_position( parameters["colsample_bylevel"], next_chunk )
    parameters["colsample_bylevel"]["values"] = get_param_values(precision,  .01, 1) 
    next_chunk = calculate_next_chunk( parameters["colsample_bylevel"], next_chunk )


    parameters["lambda"] = dict()
    parameters["lambda"]["isDefault"] = False
    parameters["lambda"]["bits"] = precision
    parameters["lambda"]["description"] = \
        "L2 regularization term on weights (analogous to Ridge regression)\
        This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often,\
        it should be explored to reduce overfitting."
    parameters["lambda"]["chromosome_position"] = dict()
    parameters["lambda"]["chromosome_position"]["start"] = get_start_position( parameters["lambda"], next_chunk)
    parameters["lambda"]["chromosome_position"]["end"] = get_end_position( parameters["lambda"], next_chunk )
    parameters["lambda"]["values"] = get_param_values(precision,  .01, 10) 
    next_chunk = calculate_next_chunk( parameters["lambda"], next_chunk )


    parameters["alpha"] = dict()
    parameters["alpha"]["isDefault"] = False
    parameters["alpha"]["bits"] = precision
    parameters["alpha"]["description"] = \
        "L1 regularization term on weight (analogous to Lasso regression)\
        Can be used in case of very high dimensionality so that the algorithm runs faster when implemented"
    parameters["alpha"]["chromosome_position"] = dict()
    parameters["alpha"]["chromosome_position"]["start"] = get_start_position( parameters["alpha"], next_chunk)
    parameters["alpha"]["chromosome_position"]["end"] = get_end_position( parameters["alpha"], next_chunk )
    parameters["alpha"]["values"] = get_param_values(precision,  .01, 10) 
    next_chunk = calculate_next_chunk( parameters["alpha"], next_chunk )


    parameters["scale_pos_weight"] = dict()
    parameters["scale_pos_weight"]["isDefault"] = False
    parameters["scale_pos_weight"]["bits"] = precision
    parameters["scale_pos_weight"]["description"] = \
        "A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence"
    parameters["scale_pos_weight"]["chromosome_position"] = dict()
    parameters["scale_pos_weight"]["chromosome_position"]["start"] = get_start_position( parameters["scale_pos_weight"], next_chunk)
    parameters["scale_pos_weight"]["chromosome_position"]["end"] = get_end_position( parameters["scale_pos_weight"], next_chunk )
    parameters["scale_pos_weight"]["values"] = get_param_values(precision,  0, .5) 
    next_chunk = calculate_next_chunk( parameters["scale_pos_weight"], next_chunk )


    #This parameters have to be set by user
    parameters["objective"] = dict()
    parameters["objective"]["isDefault"] = True
    parameters["objective"]["bits"] = 0
    parameters["objective"]["description"] = \
        "This defines the loss function to be minimized. Mostly used values are:\
        binary:logistic –logistic regression for binary classification, returns predicted probability (not class)\
        multi:softmax   –multiclass classification using the softmax objective.\
                         Returns predicted class (not probabilities)\
                         you also need to set an additional num_class (number of classes) \
                         parameter defining the number of unique classes\
        multi:softprob  –same as softmax, but returns predicted probability of each data point\
                         belonging to each class."
    parameters["objective"]["chromosome_position"] = dict()
    parameters["objective"]["chromosome_position"]["start"] = get_start_position( parameters["objective"], next_chunk)
    parameters["objective"]["chromosome_position"]["end"] = get_end_position( parameters["objective"], next_chunk )
    parameters["objective"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["objective"], next_chunk )


    parameters["eval_metric"] = dict()
    parameters["eval_metric"]["isDefault"] = True
    parameters["eval_metric"]["bits"] = 0
    parameters["eval_metric"]["description"] = \
        "eval_metric [ default according to objective ]\
        The metric to be used for validation data.\
        The default values are rmse for regression and error for classification.\
        Typical values are:\
            rmse     – root mean square error\
            mae      – mean absolute error\
            logloss  – negative log-likelihood\
            error    – Binary classification error rate (0.5 threshold)\
            merror   – Multiclass classification error rate\
            mlogloss – Multiclass logloss\
            auc      – Area under the curve"
    parameters["eval_metric"]["chromosome_position"] = dict()
    parameters["eval_metric"]["chromosome_position"]["start"] = get_start_position( parameters["eval_metric"], next_chunk)
    parameters["eval_metric"]["chromosome_position"]["end"] = get_end_position( parameters["eval_metric"], next_chunk )
    parameters["eval_metric"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["eval_metric"], next_chunk )


    parameters["eval_metric"] = dict()
    parameters["eval_metric"]["isDefault"] = True
    parameters["eval_metric"]["bits"] = 0
    parameters["eval_metric"]["description"] = \
        "eval_metric [ default according to objective ]\
        The metric to be used for validation data.\
        The default values are rmse for regression and error for classification.\
        Typical values are:\
            rmse     – root mean square error\
            mae      – mean absolute error\
            logloss  – negative log-likelihood\
            error    – Binary classification error rate (0.5 threshold)\
            merror   – Multiclass classification error rate\
            mlogloss – Multiclass logloss\
            auc      – Area under the curve"
    parameters["eval_metric"]["chromosome_position"] = dict()
    parameters["eval_metric"]["chromosome_position"]["start"] = get_start_position( parameters["eval_metric"], next_chunk)
    parameters["eval_metric"]["chromosome_position"]["end"] = get_end_position( parameters["eval_metric"], next_chunk )
    parameters["eval_metric"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["eval_metric"], next_chunk )



    parameters["seed"] = dict()
    parameters["seed"]["isDefault"] = True
    parameters["seed"]["bits"] = 0
    parameters["seed"]["description"] = \
        "The random number seed. Can be used for generating reproducible results and also for parameter tuning."
    parameters["seed"]["chromosome_position"] = dict()
    parameters["seed"]["chromosome_position"]["start"] = get_start_position( parameters["seed"], next_chunk)
    parameters["seed"]["chromosome_position"]["end"] = get_end_position( parameters["seed"], next_chunk )
    parameters["seed"]["values"] = 0
    next_chunk = calculate_next_chunk( parameters["seed"], next_chunk )

    print_hyperparameter(parameters)
    genoma_size = next_chunk

return parameters, genoma_size


