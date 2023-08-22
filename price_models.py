import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf
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


N =1


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class LINEAR_STATMODELS:
    """Linear models from statmodels library"""
    def __init__(self, FORMULA, N): #TO DO: This is the only function in price_models that has FORMULA
        self.model = FORMULA
        self.name = "linear_statmodel" 
        self.FORMULA = FORMULA
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))

        print("Lightgbm Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, FORMULA ):
        if FORMULA == "":
            print("ERROR: You have not defined FORMULA in LINEAR_STATMODELS")
        else:
            train_data= self.model.Dataset(X_train, label=y_train)
            linear_statmodel_eval  = self.model.Dataset(X_test, y_test, reference=train_data)
            error_dict = {"MSE":"l2", "R2":{"l1","l2"}, "MAE":"l1","LOGLOSS": "multi_logloss" }
            error_metric = error_dict[error_type]

            self.model = smf.ols( formula = self.FORMULA , data = train_data ).fit() # TODO:  ingresar la funcion de perdida para statsmodels

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)

    def set_formula(FORMULA):
        self.FORMULA =  FORMULA


class LGBM:
    """docstring for ClassName"""
    def __init__(self, lgb = lgb, N= 1):
        self.model = lgb
        self.name ="lightgbm"
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("Lightgbm Cores: ")

    def fit(self, X_train, y_train, X_test, y_test):
        train_data= self.model.Dataset(X_train, label=y_train)
        lgb_eval  = self.model.Dataset(X_test, y_test, reference=train_data)

        error_dict = {"MSE":"l2", "R2":{"l1","l2"}, "MAE":"l1","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model= self.model.train( 
                {'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': error_metric,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0}, 
                train_data,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=20)

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)


class ADABOOST():
    """docstring for ClassName"""
    def __init__(self, AdaBoostRegressor = AdaBoostRegressor, N = 1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "AdaBoostRegressor"
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
    def __init__(self, ExtraTreesRegressor = ExtraTreesRegressor, N = 1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name =  "ExtraTreesRegressor",
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


class XGBOOST:
    """docstring for ClassName"""
    def __init__(self, XGBRegressor = XGBRegressor, N=1):
        self.model = XGBRegressor
        self.name = "XGBRegressor"
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        print("XGBoostRegressor Cores: ", self.cores_number )

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):

        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]

        self.model = XGBRegressor(
                    max_depth=14, 
                    learning_rate=0.05, 
                    n_estimators=800, 
                    silent=True, 
                    objective='reg:linear', 
                    nthread=8, 
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

        self.model.fit(X_train, y_train, eval_metric=error_metric, 
            verbose = True) #eval_set = [(X_train, y_train), (X_test, y_test)], #<- Verbose, early_stopping_rounds=20)


    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)


class BAGGING():
    """docstring for ClassName"""
    def __init__(self, BaggingRegressor=BaggingRegressor, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "BaggingRegressor"
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


class LINEARREGRESSION:
    """docstring for ClassName"""
    def __init__(self, LinearRegression = LinearRegression, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "LinearRegression"
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
    def __init__(self, ARDRegression=ARDRegression, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "ARDRegression"
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
    def __init__(self, BayesianRidge = BayesianRidge, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "BayesianRidge"
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
    def __init__(self, ElasticNet = ElasticNet, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "ElasticNet"
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
    def __init__(self, RandomForestRegressor=RandomForestRegressor, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "RandomForestRegressor"
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
    """This mdoel is extremly expensive since it does't allow parallelization"""
    def __init__(self, GradientBoostingRegressor=GradientBoostingRegressor, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "GradientBoostingRegressor"
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
    def __init__(self, MultinomialNB=MultinomialNB, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "MultinomialNB"
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
    def __init__(self, GaussianNB=GaussianNB, N=1):
        self.cores_number = int(np.ceil(multiprocessing.cpu_count()/N))
        self.name = "GaussianNB"
        self.model = GaussianNB(priors=None)

        print("GaussianNB Cores: ", np.nan)

    def fit(self, X_train, y_train, X_test, y_test, error_type = "MAE"):
        error_dict = {"MSE":"rmse", "R2":{"l1","l2"}, "MAE":"mae","LOGLOSS": "multi_logloss" }
        error_metric = error_dict[error_type]
        self.model.fit(X_train, y_train )

    def predict(self, X_test):
         prediction=self.model.predict(X_test)
         return(prediction)




ERROR_TYPES = {'MAE': mean_absolute_error,
 'MSE': mean_squared_error,
 'R2': r2_score}

save_obj(ERROR_TYPES, "ERROR_TYPES")
