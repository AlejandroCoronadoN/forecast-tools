import pandas as pd 
import numpy as np 
import multiprocessing
from xgboost import XGBRegressor
from xgboost import plot_importance
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
import lightgbm as lgb 
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
                             BaggingRegressor, ExtraTreesRegressor, \
                             GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, BayesianRidge,ElasticNet, ARDRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import importlib 
import aiqutils.feature_selection


N =1


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class TEST_PARALLEL():
    """docstring for ClassName"""
    def __init__(self, model, N):
        self.model = ""
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print(" TEST MODEL ")

    def fit(self, X_train, y_train, X_test, y_test):
        self.hash_columns =hash("".join([str(x) for x in list(X_train.columns)]))

    def predict(self, X_test):
        #print(self.hash_columns)
        return(self.hash_columns)


class LGBM():
    """docstring for ClassName"""
    def __init__(self, lgb, N):
        self.model = ""
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("Lightgbm Cores: ")

    def fit(self, X_train, y_train, X_test, y_test,error_type = "MAE"):

        error_dict = {"MSE":"l2", "R2":{"l1","l2"}, "MAE":"l1","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model= lgb.LGBMRegressor(
                        num_leaves=31,
                        learning_rate=0.07,
                        n_estimators=100,
                        subsample=.9,
                        colsample_bytree=.9,
                        random_state=1 )
        self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=100,
                eval_metric=error_metric)

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)






class ADABOOST():
    """docstring for ClassName"""
    def __init__(self, AdaBoostRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = AdaBoostRegressor(
            base_estimator=None, 
            learning_rate=1.0, 
            loss='linear',
            n_estimators=50, 
            random_state=None)

        print("AdaBoostRegressor Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class EXTRATREE():
    """docstring for ClassName"""
    def __init__(self, ExtraTreesRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = ExtraTreesRegressor(
                      bootstrap=False, 
                      criterion='mse', 
                      max_depth=None,
                      max_features='auto', 
                      max_leaf_nodes=None,
                      min_impurity_decrease=0.0, 
                      min_impurity_split=None,
                      min_samples_leaf=1, 
                      min_samples_split=2,
                      min_weight_fraction_leaf=0.0, 
                      n_estimators=200, 
                      n_jobs=self.cores_number,
                      oob_score=False, 
                      random_state=None, 
                      verbose= True, 
                      warm_start=False)


        print("ExtraTreesRegressor Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"mse", "MAE":"mae" }
        error_metric = error_dict[error_type]
        self.model.set_params(criterion = error_metric)
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class XGBOOST():
    """docstring for ClassName"""
    def __init__(self, precision = None, XGBRegressor = XGBRegressor):
        self.model = XGBRegressor(
            eta= 0.1,
            nthread = 1,
            max_leaf_nodes = 1, 
            reg_lambda = 1,
            reg_alpha = 1,
            eval_metric =  "mae")


        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("XGBoostRegressor Cores: ", self.cores_number )
        self.hyperparameters = None
        if precision:
            self.precision = precision
            self.hyperparameters_dictionary, self.hyperparameter_chromosome_size = self._get_xgboost_hyperhyperparameters(precision)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        if self.hyperparameters:
            self.model.set_params(**self.hyperparameters)
        else:
            #Default training model
            self.model = XGBRegressor(max_depth=6,  #Equals to no limit equivalent to None sklearn
                            learning_rate=0.075, 
                            n_estimators=100, 
                            silent=True, 
                            objective='reg:linear', 
                            nthread=1, 
                            gamma=0,
                            min_child_weight=1, 
                            max_delta_step=0, 
                            subsample=0.85, 
                            colsample_bytree=0.7, 
                            colsample_bylevel=1, 
                            reg_alpha=0, 
                            reg_lambda=1,   
                            scale_pos_weight=1, 
                            seed=1440, 
                            missing=None)

        self.model.fit(X_train, y_train, 
            eval_metric=error_metric, verbose = False, 
            eval_set = [(X_train, y_train), (X_test, y_test)], 
            early_stopping_rounds=20)


    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)

    def _update_hyperparameters(self, hyperparameter_chromosome, default_param_dict):
        """ Upadtes the hyperparamters after there is a change in the hyperparameter_chromosome. 
        There is a relation between each hyperparameter and the hyperparameter_chromosome. For each hyperparameter
        there is a chunk inside the chromosome  that determines the value that this hyperparameter will take.
        This function just converts the binary represnetation in the chunk and select the value for that hyperparameter
        using a list of all the posible values that the hyperparameter can take.  
        """
        if len(hyperparameter_chromosome) != self.hyperparameter_chromosome_size:
            raise ValueError("Error: hyperparameter_chromosome is not the same size as self.hyperparameter_chromosome_size")
        else:
            hyperparameters = dict() 
            for key in self.hyperparameters_dictionary.keys():
                if self.hyperparameters_dictionary[key]["isDefault"] == True and key not in default_param_dict.keys():
                    hyperparameters[key] = self.model.get_params()[key]
                    continue

                if key in default_param_dict.keys():
                    self.hyperparameters_dictionary[key]["selected_value"] =  default_param_dict[key]
                    hyperparameters[key] =  default_param_dict[key]
                    
                else:
                    chunk = hyperparameter_chromosome[self.hyperparameters_dictionary[key]["chromosome_position"]["start"] : self.hyperparameters_dictionary[key]["chromosome_position"]["end"]]
                    value_position = int(chunk, 2)
                    value = self.hyperparameters_dictionary[key]["values"][value_position]
                    hyperparameters[key] = value
                    self.hyperparameters_dictionary[key]["selected_value"] = value
            self.hyperparameters =  hyperparameters
            self.model.set_params(**self.hyperparameters)
    


    def _get_xgboost_hyperhyperparameters(self, precision):
        """
        Creates a dictionary with all the hyperparameters of XGBoost model. 
        This dictionary is used in the
        """

        next_chunk = 0
        parameters = dict()
        parameters["booster"] = dict()
        parameters["booster"]["isDefault"] = True
        parameters["booster"]["bits"] = 1
        parameters["booster"]["description"] = \
            "booster [default=gbtree] Select the type of model to run at each iteration. It has 2 options:\
            gbtree: tree-based models gblinear: linear models <----- This kind of booster is never used."
        parameters["booster"]["chromosome_position"] = dict()
        parameters["booster"]["chromosome_position"]["start"] = get_start_position( parameters["booster"] , next_chunk)
        parameters["booster"]["chromosome_position"]["end"] = get_end_position( parameters["booster"] , next_chunk)
        parameters["booster"]["values"] = ["gbtree", "gblinear"]
        parameters["booster"]["selected_value"] = np.nan
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
        parameters["silent"]["selected_value"] = np.nan
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
        parameters["nthread"]["selected_value"] = np.nan
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
        parameters["eta"]["values"] = get_param_values(precision, .01, .7, False)
        parameters["eta"]["selected_value"] = np.nan
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
        parameters["min_child_weight"]["values"] = get_param_values(precision, 1, False, True)
        parameters["min_child_weight"]["selected_value"] = np.nan
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
        parameters["max_depth"]["values"] = get_param_values(precision, 1, False, True)
        parameters["max_depth"]["selected_value"] = np.nan
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
        parameters["max_leaf_nodes"]["values"] = get_param_values(precision, 1, False, True)
        parameters["max_leaf_nodes"]["selected_value"] = np.nan
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
        parameters["gamma"]["selected_value"] = 0
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
        parameters["max_delta_step"]["values"] = get_param_values(precision,  5, False, False) #by definition
        parameters["max_delta_step"]["selected_value"] = np.nan
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
        parameters["subsample"]["values"] = get_param_values(precision,  .1, 1, False) 
        parameters["subsample"]["selected_value"] = np.nan
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
        parameters["colsample_bytree"]["selected_value"] = np.nan
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
        parameters["colsample_bylevel"]["selected_value"] = np.nan
        next_chunk = calculate_next_chunk( parameters["colsample_bylevel"], next_chunk )


        parameters["reg_lambda"] = dict()
        parameters["reg_lambda"]["isDefault"] = False
        parameters["reg_lambda"]["bits"] = precision
        parameters["reg_lambda"]["description"] = \
            "L2 regularization term on weights (analogous to Ridge regression)\
            This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often,\
            it should be explored to reduce overfitting."
        parameters["reg_lambda"]["chromosome_position"] = dict()
        parameters["reg_lambda"]["chromosome_position"]["start"] = get_start_position( parameters["reg_lambda"], next_chunk)
        parameters["reg_lambda"]["chromosome_position"]["end"] = get_end_position( parameters["reg_lambda"], next_chunk )
        parameters["reg_lambda"]["values"] = get_param_values(precision,  .01, 10) 
        parameters["reg_lambda"]["selected_value"] = np.nan
        next_chunk = calculate_next_chunk( parameters["reg_lambda"], next_chunk )


        parameters["reg_alpha"] = dict()
        parameters["reg_alpha"]["isDefault"] = False
        parameters["reg_alpha"]["bits"] = precision
        parameters["reg_alpha"]["description"] = \
            "L1 regularization term on weight (analogous to Lasso regression)\
            Can be used in case of very high dimensionality so that the algorithm runs faster when implemented"
        parameters["reg_alpha"]["chromosome_position"] = dict()
        parameters["reg_alpha"]["chromosome_position"]["start"] = get_start_position( parameters["reg_alpha"], next_chunk)
        parameters["reg_alpha"]["chromosome_position"]["end"] = get_end_position( parameters["reg_alpha"], next_chunk )
        parameters["reg_alpha"]["values"] = get_param_values(precision,  .01, 10) 
        parameters["reg_alpha"]["selected_value"] = np.nan
        next_chunk = calculate_next_chunk( parameters["reg_alpha"], next_chunk )


        parameters["scale_pos_weight"] = dict()
        parameters["scale_pos_weight"]["isDefault"] = False
        parameters["scale_pos_weight"]["bits"] = precision
        parameters["scale_pos_weight"]["description"] = \
            "A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence"
        parameters["scale_pos_weight"]["chromosome_position"] = dict()
        parameters["scale_pos_weight"]["chromosome_position"]["start"] = get_start_position( parameters["scale_pos_weight"], next_chunk)
        parameters["scale_pos_weight"]["chromosome_position"]["end"] = get_end_position( parameters["scale_pos_weight"], next_chunk )
        parameters["scale_pos_weight"]["values"] = get_param_values(precision,  0, .5) 
        parameters["scale_pos_weight"]["selected_value"] = np.nan
        next_chunk = calculate_next_chunk( parameters["scale_pos_weight"], next_chunk )


        #This parameters have to be set by user
        parameters["objective"] = dict()
        parameters["objective"]["isDefault"] = True
        parameters["objective"]["bits"] = 0
        parameters["objective"]["description"] = \
            "This defines the loss function to be minimized. Mostly used values are:\
            binary:logistic - logistic regression for binary classification, returns predicted probability (not class)\
            multi:softmax   - Multiclass classification using the softmax objective.\
                              Returns predicted class (not probabilities)\
                              you also need to set an additional num_class (number of classes) \
                              parameter defining the number of unique classes\
            multi:softprob  - same as softmax, but returns predicted probability of each data point\
                            - belonging to each class.\
            reg:linear:     - linear regression\
            reg:logistic:   - logistic regressi"
        parameters["objective"]["chromosome_position"] = dict()
        parameters["objective"]["chromosome_position"]["start"] = get_start_position( parameters["objective"], next_chunk)
        parameters["objective"]["chromosome_position"]["end"] = get_end_position( parameters["objective"], next_chunk )
        parameters["objective"]["values"] = 0
        parameters["objective"]["selected_value"] = np.nan
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
        parameters["eval_metric"]["selected_value"] = np.nan
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
        parameters["seed"]["selected_value"] = np.nan
        next_chunk = calculate_next_chunk( parameters["seed"], next_chunk )

        print_hyperparameter(parameters)
        genoma_size = next_chunk

        return parameters, genoma_size





class BAGGING():
    """docstring for ClassName"""
    def __init__(self, BaggingRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = BaggingRegressor(
                 base_estimator=None, 
                 bootstrap=True,
                 bootstrap_features=False, 
                 max_features=1.0, 
                 max_samples=1.0,
                 n_estimators=10, 
                 n_jobs= self.cores_number, 
                 oob_score=False, 
                 random_state=None,
                 verbose=0, 
                 warm_start=False)


        print("Bagging Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)



class LINEARREGRESSION():
    """docstring for ClassName"""
    def __init__(self, LinearRegression, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = LinearRegression(n_jobs = self.cores_number, normalize = False)
        print("LinearRegression Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class ARDR():
    """docstring for ClassName"""
    def __init__(self, ARDRegression, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.selected_columns = []
        self.model = ARDRegression(
                        alpha_1=1e-06, 
                        alpha_2=1e-06, 
                        compute_score=False, 
                        copy_X=True,
                        fit_intercept=True, 
                        lambda_1=1e-06, 
                        lambda_2=1e-06, 
                        n_iter=300,
                        normalize=False, 
                        threshold_lambda=10000.0, 
                        tol=0.001, verbose=False)


        print("ARDRegression Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        try:
            self.selected_columns = np.random.choice(X_train.columns, 100, replace = False)
            X_train = X_train[self.selected_columns]
        except Exception as E:
            X_train = X_train
              
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test[self.selected_columns])
         return(prediction)




class BAYESIANRIDGE():
    """docstring for ClassName"""
    def __init__(self, BayesianRidge, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = BayesianRidge(
                        alpha_1=1e-06, 
                        alpha_2=1e-06, 
                        compute_score=False, 
                        copy_X=True,
                        fit_intercept=True, 
                        lambda_1=1e-06, 
                        lambda_2=1e-06, 
                        n_iter=300,
                        normalize=False, 
                        tol=0.001, 
                        verbose=False)


        print("BayesianRidge Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




class ELASTIC():
    """docstring for ClassName"""
    def __init__(self, ElasticNet, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = ElasticNet(
                        alpha=1.0, 
                        copy_X=True, 
                        fit_intercept=True, 
                        l1_ratio=0.5,
                        max_iter=1000, 
                        normalize=False, 
                        positive=False, 
                        precompute=False,
                        random_state=None, 
                        selection='cyclic', 
                        tol=0.0001, 
                        warm_start=False)

        print("ElasticNet Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class RANDOMFOREST():
    """docstring for ClassName"""
    def __init__(self, RandomForestRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = RandomForestRegressor(
                       bootstrap=True, criterion='mse',
                       max_depth=None,
                       max_features='auto',
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_impurity_split=None,
                       min_samples_leaf=1,
                       min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       n_estimators=200, 
                       n_jobs=1,
                       oob_score=False, 
                       random_state=None,
                       verbose= True,
                       warm_start=False)


        print("RandomForestRegressor Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class GRADIENTBOOSTING():
    """This mdoel is extremly expensive since it does't allow apralellization"""
    def __init__(self, GradientBoostingRegressor, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        self.model = GradientBoostingRegressor(
                         criterion='friedman_mse', 
                         init=None,
                         learning_rate=0.1, 
                         max_depth=3,  #test None
                         max_features=None,
                         max_leaf_nodes=None, 
                         min_impurity_decrease=0.0,
                         min_impurity_split=None, 
                         min_samples_leaf=1,
                         min_samples_split=2, 
                         min_weight_fraction_leaf=0.0,
                         n_estimators=200,  #test 100, 50
                         presort='auto', 
                         random_state=None,
                         subsample=1.0, 
                         verbose=True, 
                         warm_start=False)

        print("GradientBoostingRegressor Cores: ", self.cores_number)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"mse", "MAE":"mae" }
        error_metric = error_dict[error_type]
        #self.model.set_params(criterion = error_metric)
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)





class MULTINOMIALNB():
    """docstring for ClassName"""
    def __init__(self, MultinomialNB, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = MultinomialNB(
                        alpha=1.0, 
                        class_prior=None, 
                        fit_prior=True)


        print("MultinomialNB Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




class GAUSSIANNB():
    """docstring for ClassName"""
    def __init__(self, GaussianNB, N):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.model = GaussianNB(priors=None)

        print("GaussianNB Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)


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

def get_param_values(precision, min_value,max_value, isInt = False):
    """ Returns a set of values that can be used in a particular parameter that lives between
    min_value and max_value.

    Input
    ----------
    precision: The number of bit that will be used to test different combiantions of values for this
               hyperparameter num of values = 2^precision
    min_value, max_value: range(min_value, max_value)
    isInt: Set True if the output values for a particular hyperparameter require to be integers.

    Output
    ----------
    A renage of values that can be used for a particular hyperparameter.
    """
    if max_value:
        precision_pow2 = np.power(2, precision)
        steps = (max_value - min_value)/ precision_pow2
        values = np.arange(min_value, max_value, steps)
        values[len(values)-1] = max_value

    else:
        precision_pow2 = np.power(2, precision)
        max_value = precision_pow2 + min_value
        steps = (max_value - min_value)/ precision_pow2
        values = np.arange(min_value, max_value, steps)
        values[len(values)-1] = max_value

    if isInt:        
        print("\n\nTesting:", values)
        values = [int(np.round(x)) for x in values]
        print("Testing:", values)




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

