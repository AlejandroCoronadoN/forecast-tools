import pandas as pd 
import numpy as np 
import random 
from   pprint import pprint 
import copy
import pdb, traceback, sys
import multiprocessing
import random
import time
from sklearn import linear_model
import pickle
import threading

def generate_model_population(N, model, default_param_dict):
    """ Generates N individuals each onw representing the same baseline model but with different 
    hyperparamters.

    Input
    ----------
    N: number of individuals
    model: model to be implemented (it must be a model from aiqutils library with hyperparameter configurable
    dictionary)
    default_param_dict: hyperparameters cant be set randomly (seed, eta..)
    
    Output
    ----------
    Dictioanry of length 10 in which each entrance represent a different individual.
    """
    POPULATION = dict()
    hyperparameters_dictionary = model.hyperparameters_dictionary
    hyperparameter_chromosome_size = model.hyperparameter_chromosome_size

    for i in range(N):
        model_i = copy.deepcopy(model)
        POPULATION[i] = dict()
        hyperparameters_chromosome = list("0"*hyperparameter_chromosome_size)

        for j in  range(len(hyperparameters_chromosome)):
            is1 = random.choice([True, False])
            if is1:
                hyperparameters_chromosome[j] = "1"
        hyperparameters_chromosome = "".join( hyperparameters_chromosome)

        model_i._update_hyperparameters(hyperparameters_chromosome, default_param_dict)

        hyperparameters = model_i.hyperparameters
        POPULATION[i]["model"] =  model_i
        POPULATION[i]["hyperparameters"] = hyperparameters
        print("\n\nTesting hyperparameters: ", i, "\n\t", hyperparameters)
        POPULATION[i]["hyperparameters_chromosome"] = hyperparameters_chromosome
        POPULATION[i]["SCORE"]  = np.nan

    return POPULATION


def solve_genetic_algorithm(N, PC, PM, N_WORKERS, MAX_ITERATIONS, model, default_param_dict,
    df_kfolded, max_features=1000, round_prediction= False, parallel_execution = False ):
    """
    Solve a Genetic Algorithm with 
        len(genoma)               == GENLONG, 
        POPULATION                == N individuals
        Mutation probability      == PM
        Crossover probability     == PC
        Max number of iterations  == MAX_ITERATIONS
    """

    print("\n\nTEST KFOLDED MEANS: ")
    for fold in df_kfolded.keys():
        print("FOLD", np.mean(df_kfolded[fold]["y"]))

    STILL_CHANGE = True 
    still_change_count = 0 

    if parallel_execution:
        LOCK = multiprocessing.Lock()
        manager_solutions = multiprocessing.Manager()
        SOLUTIONS = manager_solutions.dict()
    else: 
        SOLUTIONS = dict()

    POPULATION_X = generate_model_population(N, model, default_param_dict)

    if parallel_execution:
        print("\n\n----------------PARALLEL SOLVE--------------")
        POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS,
                                      df_kfolded, "MSE", max_features, round_prediction)
    else:
        print("\n\n--------------SEQUENTIAL SOLVE----------------")
        POPULATION_X = sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS,
                                      df_kfolded, "MSE", max_features, round_prediction)

    POPULATION_X = sort_population(POPULATION_X)
    n_iter = 0 
    while (n_iter <= MAX_ITERATIONS) & STILL_CHANGE:

        POPULATION_Y = cross_mutate( POPULATION_X, N, PC, PM, default_param_dict)
        print("\n\n******************** test_chromosome_xy after cross_mutate")
        test_chromosome_xy(POPULATION_Y, POPULATION_X)

        if parallel_execution:
            print("\n\n----------------PARALLEL SOLVE--------------\n POPULATION_X length: {} POPULATION_Y length{}".format(len(POPULATION_X.keys()), len(POPULATION_Y.keys())))
            POPULATION_X = parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, 
                           df_kfolded, "MSE", max_features, round_prediction)
            POPULATION_Y = parallel_solve(POPULATION_Y, SOLUTIONS, LOCK, N_WORKERS,
                           df_kfolded, "MSE", max_features, round_prediction)

        else:
            print("\n\n--------------SEQUENTIAL SOLVE----------------\n POPULATION_X: {} POPULATION_Y {}".format(len(POPULATION_X.keys()), len(POPULATION_Y.keys())))
            POPULATION_X = sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS, 
                           df_kfolded, "MSE", max_features, round_prediction)
            POPULATION_Y = sequential_solve(POPULATION_Y, SOLUTIONS, N_WORKERS,
                           df_kfolded, "MSE", max_features, round_prediction)
            #print("\n\ntest_chromosome_xy 1")
            #test_chromosome_xy(POPULATION_X, POPULATION_Y)
        print("\n\n2 -----------------TEST: fenotye in SOLUTIONS acordingly to genoma")
        test_fenotype_chromosome_SOLUTIONS(SOLUTIONS)
        #print("\n\ntest_chromosome_xy 2")
        #test_chromosome_xy(POPULATION_X, POPULATION_Y)

        POPULATION_X = sort_population(POPULATION_X)
        POPULATION_Y = sort_population(POPULATION_Y)
        POPULATION_X = select_topn(POPULATION_X, POPULATION_Y, N)

        #print("\n\ntest_chromosome_xy 3")
        #test_chromosome_xy(POPULATION_X, POPULATION_X)

        equal_individuals, max_score = population_summary(POPULATION_X)

        n_iter += 1
        if equal_individuals >= len(POPULATION_X.keys())*.8:
            still_change_count +=1 
            #print("SAME ** {} ".format(still_change_count))
            if still_change_count >= 10:
                print("\n\n\nGA Solved: \n\tN: {}\n\tPC:{}, \n\tPM: {}\n\tN_WORKERS: {} \n\tMAX_ITERATIONS: {}".format( N, PC, PM, N_WORKERS, MAX_ITERATIONS ))
                STILL_CHANGE = False

    return POPULATION_X, SOLUTIONS


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


def test_chromosome_xy(POPULATION_X, POPULATION_Y):
    equal_individuals = 0
    for key in POPULATION_Y.keys():
        hyperparameters_chromosome_y =  POPULATION_Y[key]["hyperparameters_chromosome"]
        hyperparameters_chromosome_x =  POPULATION_X[key]["hyperparameters_chromosome"]
        if hyperparameters_chromosome_x == hyperparameters_chromosome_y:
           equal_individuals += 1 
    if equal_individuals == len(POPULATION_X.keys()):
        print("\t WARNING: hyperparameters_chromosome are the same in POPULATION_X and POPULATION_Y")
    print("\t BASELINE: equal_individuals in POPULATION_X and POPULATION_Y: {} out of X{}, Y{} ".format( equal_individuals, len(POPULATION_Y), len(POPULATION_X)))



def parallel_solve(POPULATION_X, SOLUTIONS, LOCK, N_WORKERS, df_kfolded, 
    error_type, max_features, round_prediction):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function.
    """
    parallel_execution = True
    s = 0
    POPULATION_SIZE = len(POPULATION_X)
    while s < POPULATION_SIZE-1: 
        #EN esta parte antes de meterlo a procesamiento deberia buscar la solucion...
        started_process = list()
        active_workers = 0

        while active_workers < N_WORKERS:
            if s >= POPULATION_SIZE:
                break
            else:
                if POPULATION_X[s]["hyperparameters_chromosome"] in SOLUTIONS.keys():
                    s+=1
                else:
                    #<------SOLVE
                    #TO DO: tenemos Z started_processes, estos deberian repartire los cores
                    started_process.append(s)
                    s+=1
                    active_workers +=1
        for s in started_process:
            time.sleep( random.randint(0,len(started_process)) * .05)
            INDIVIDUAL = copy.deepcopy(POPULATION_X[s])
            POPULATION_X[s]["PROCESS"] = multiprocessing.Process( target= score_model , 
                                        args=(INDIVIDUAL, SOLUTIONS, LOCK, df_kfolded, error_type,
                                        max_features, round_prediction, parallel_execution, ))

            POPULATION_X[s]["PROCESS"].start()

        #print("STARTED PROCESSES: \n\t", started_process)

        for sp in started_process:  # <---CUANTOS CORES TENGO 
            POPULATION_X[sp]["PROCESS"].join()
            del POPULATION_X[sp]["PROCESS"]

    #save_obj(SOLUTIONS, "ga_solutions")
    #All genoma in SOLUTIONS
    for s in POPULATION_X.keys():
        if  POPULATION_X[s]["hyperparameters_chromosome"] in SOLUTIONS.keys():
            POPULATION_X[s]["SCORE"] = SOLUTIONS[POPULATION_X[s]["hyperparameters_chromosome"]]["score"]
            POPULATION_X[s]["PROB"] = POPULATION_X[s]["SCORE"] / POPULATION_SIZE
        else:
                print("**WARNING: hyperparameters_chromosome not found in SOLUTIONS after scoring")

    return POPULATION_X


def sequential_solve(POPULATION_X, SOLUTIONS, N_WORKERS, df_kfolded, 
                    error_type, max_features, round_prediction):
    """
    Each individual in POPULATION_X is assigned to a different process to score it's own
    genoma. Each genoma is scored based on the scoring function.
    """
    s = 0
    POPULATION_SIZE = len(POPULATION_X)
    for s in POPULATION_X.keys():
        if POPULATION_X[s]["hyperparameters_chromosome"] in SOLUTIONS.keys():
            continue
        else:
            INDIVIDUAL = copy.deepcopy(POPULATION_X[s])
            score_model(INDIVIDUAL, SOLUTIONS, None, df_kfolded, 
                    error_type, max_features, round_prediction, parallel_execution = False)

    for s in POPULATION_X.keys():
        if  POPULATION_X[s]["hyperparameters_chromosome"] in SOLUTIONS.keys():
            POPULATION_X[s]["SCORE"] = SOLUTIONS[POPULATION_X[s]["hyperparameters_chromosome"]]["score"]
            POPULATION_X[s]["PROB"] = POPULATION_X[s]["SCORE"] / POPULATION_SIZE
        else:
            print("**WARNING: hyperparameters_chromosome not found in SOLUTIONS after scoring")

    return POPULATION_X


def score_model(INDIVIDUAL, SOLUTIONS, LOCK, df_kfolded, error_type, 
    max_features, round_prediction = False, parallel_execution = True):
    """
    Scores the model inside a neuron given a particular set of variables and calculates
    the ERROR_TYPES[error_type] for each fold in df_kfolded.
    """
    hyperparameters =  INDIVIDUAL["hyperparameters"]
    hyperparameters_chromosome = INDIVIDUAL["hyperparameters_chromosome"]

    model = copy.deepcopy(INDIVIDUAL["model"]) #XGBOOST(XGBRegressor, 1)
    total_error = 0

    for test_fold in df_kfolded.keys():
        #print("test_fold: ", test_fold)
        if test_fold == "all_data":
            continue
        else:
            X_test  = df_kfolded[test_fold]["data"].copy()
            y_test  = df_kfolded[test_fold]["y"].copy()

            X_train = pd.DataFrame()
            y_train = pd.DataFrame()

            for train_fold in df_kfolded.keys():
                #print("train_fold: ", train_fold)
                if train_fold == "all_data" or train_fold == test_fold:
                    continue 
                else:
                    if len(X_train)==0:
                        X_train = df_kfolded[train_fold]["data"]
                        y_train = df_kfolded[train_fold]["y"].copy()

                    else:
                        X_train_append = df_kfolded[train_fold]["data"]
                        X_train = X_train.append(X_train_append)
                        y_train = y_train.append(df_kfolded[train_fold]["y"].copy()) 

            model.fit(X_train, y_train, X_test, y_test)
            prediction   = model.predict(X_test)
            prediction[prediction < 0] = 0
            y_test = np.log1p(y_test)

            error = error_function(y_test, prediction)
            total_error += error

    score =  - (total_error/(len(df_kfolded.keys())-1))
    genoma_solutions = dict()
    genoma_solutions["score"] = score
    genoma_solutions["hyperparameters"] = hyperparameters
    genoma_solutions["model"] = model
    genoma_solutions["hyperparameters_chromosome"] = hyperparameters_chromosome

    if parallel_execution:
        time.sleep( random.randint(0,16) * .05)
        LOCK.acquire()
        SOLUTIONS[hyperparameters_chromosome] = genoma_solutions
        LOCK.release()

    else:
        SOLUTIONS[hyperparameters_chromosome]= genoma_solutions


def test_wrongfold_df_kfolded(df_kfolded):
    flag = False
    for fold in df_kfolded.keys():
        data_fold = df_kfolded[fold]["data"]
        if fold == "all_data":
            duplicates_alldata = np.sum(df_kfolded[fold]["data"].duplicated())
        else:
            other_data = pd.DataFrame()
            for fold2 in df_kfolded.keys():
                if fold2 != fold and fold2 != "all_data":
                    if len(other_data) ==0:
                        other_data = df_kfolded[fold2]["data"]
                    else :
                        other_data = other_data.append(df_kfolded[fold2]["data"])
                else:
                    continue
            duplicates_otherdata =  np.sum(other_data.duplicated())
            duplicates_inside_fold = np.sum(data_fold.duplicated())
            all_data = data_fold.append(other_data)
            duplicates_alldata =  np.sum(all_data.duplicated())
            if duplicates_alldata > 0 :
                print("\n\nWARNING: Duplicates found at fold {}: \n\tother_data:{}  inside_fold: {} all_data pct{}".format(fold, duplicates_otherdata, duplicates_inside_fold, duplicates_alldata/len(all_data) ))
                flag = True
    if not flag:
        print("\t **test_wrongfold_df_kfolded passed !")

def test_wrongfold_assignation(X_train, X_test):
    X = X_train.append(X_test)
    try:
        duplicates= np.sum(X.duplicated())
        if duplicates >0:
            print("WARNING: duplicated values in test set -  ", duplicates)
    except Exception as e:
        print(e)


def error_function(y_test, prediction):
    difference =  y_test -prediction
    difference_abs =  np.abs(difference)
    error  = difference_abs.mean()
    return error


def save_solutions(genoma, new_score, SOLUTIONS):
    SOLUTIONS[genoma]["score"] = new_score


def sort_population(POPULATION_X):
    POPULATION_X_copy = copy.deepcopy(POPULATION_X)
    POPULATION_sorted = sorted(POPULATION_X_copy.items(), key=lambda x: x[1]["SCORE"], reverse = True)
    POPULATION_NEW = dict()
    result = list()
    cont = 0
    for i in POPULATION_sorted:
        POPULATION_NEW[cont] = i[1]
        cont += 1
    #test_chromosome_xy(POPULATION_X, POPULATION_NEW)
    return POPULATION_NEW

def population_summary(POPULATION_X):
    POPULATION_X_copy = POPULATION_X.copy()
    suma = 0
    min_score =  100000000000000000000000
    max_score = -100000000000000000000000
    lista_genomas = list()
    hyperparameters_chromosome = []
    for key in POPULATION_X_copy.keys():
        individual_score =  POPULATION_X_copy[key]["SCORE"]
        lista_genomas.append(POPULATION_X[key]["hyperparameters_chromosome"])
        suma += individual_score
        if max_score < individual_score:
            max_score = individual_score
            hyperparameters_chromosome = POPULATION_X_copy[key]["hyperparameters_chromosome"]
        if min_score > individual_score:
            min_score = individual_score
    promedio = suma/len(POPULATION_X_copy.keys())
    equal_individuals = len(lista_genomas) - len(set(lista_genomas))
    print("\n\nTEST population_summary: \
        \n\tTOTAL SCORE: {} \n\tMEAN SCORE: {} \
        \n\t MAX_SCORE: {} \n\tMIN_SCORE: {} \
        \n\t BASELINE FEAT: {} ".format(suma, promedio, max_score,\
         min_score, hyperparameters_chromosome))

    return equal_individuals, max_score


def select_topn( POPULATION_X, POPULATION_Y, N):
    POPULATION_Y_copy =  copy.deepcopy(POPULATION_Y)
    POPULATION_X_copy = copy.deepcopy(POPULATION_X)
    for key in POPULATION_X_copy.keys():
        new_key = key  + N
        POPULATION_Y_copy[new_key] = POPULATION_X_copy[key]

    POPULATION_Y_copy = sort_population(POPULATION_Y_copy)
    POPULATION_NEW =dict()

    for key in range(N):
        POPULATION_NEW[key] = POPULATION_Y_copy[key]

    return POPULATION_NEW

def _mutation_per_bit(genoma, PM):
    """ This function applies a mutation techique for all bits for all the
    indiviuals in the POPULATION.
    """
    mutated_genoma = genoma
    for mutation_gen in range(len(genoma)):
        pm = random.uniform(1,0)
        if pm < PM:
            start = mutated_genoma
            if mutated_genoma[mutation_gen] =="1":
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "0"
                mutated_genoma = "".join(mutated_genoma)
            else:
                mutated_genoma = list(mutated_genoma)
                mutated_genoma[mutation_gen] = "1"
                mutated_genoma = "".join(mutated_genoma)
    return mutated_genoma


def _mutation_per_individual(genoma, PM):
    """ This function applies a mutation techique in a way that only one of the bits per indiviual
    is changed. 
    """
    mutation_gen = random.randint(0, len(genoma) -1)
    mutated_genoma  = genoma
    pm = random.uniform(1,0)

    if pm < PM:
        start = mutated_genoma
        if mutated_genoma[mutation_gen] =="1":
            mutated_genoma = list(mutated_genoma)
            mutated_genoma[mutation_gen] = "0"
            mutated_genoma = "".join(mutated_genoma)
        else:
            mutated_genoma = list(mutated_genoma)
            mutated_genoma[mutation_gen] = "1"
            mutated_genoma = "".join(mutated_genoma)
    return mutated_genoma 


def cross_mutate( POPULATION_X, N, PC, PM, default_param_dict = None, mutation_type = "mutation_per_bit"):
    POPULATION_Y = copy.deepcopy(POPULATION_X)
    #pprint(POPULATION_X)
    for j in range(int(N/2)):
        pc = random.uniform(1,0)
        GENLONG =  len(POPULATION_X[0]["hyperparameters_chromosome"])
        #CROSSOVER
        if pc < PC:
            best = POPULATION_Y[j]["hyperparameters_chromosome"]
            worst = POPULATION_Y[N -j-1]["hyperparameters_chromosome"]
            startBest =  best
            startWorst = worst

            genoma_crossover = random.randint(0, GENLONG)
            genoma_crossover_final = int(genoma_crossover + np.round(GENLONG/2))
            if genoma_crossover_final > GENLONG:
                # To perform a roulette  cross over we need to fill the
                extra_genes = genoma_crossover_final - GENLONG 
                genoma_crossover_final =  GENLONG

            else: 
                extra_genes = 0

            best_partA  = best[:extra_genes]
            worst_partA = worst[:extra_genes]

            best_partB  = best[extra_genes:genoma_crossover]
            worst_partB = worst[extra_genes:genoma_crossover]

            best_partC  = best[genoma_crossover:genoma_crossover_final]
            worst_partC = worst[genoma_crossover: genoma_crossover_final]

            best_partD  = best[genoma_crossover_final:GENLONG]
            worst_partD = worst[genoma_crossover_final: GENLONG]

            #print("TEST CROSSOVER:   {} -> {} -> {} -> {} \npart A {} \npartB {} \npart C{} \npartD {}".format(extra_genes, genoma_crossover, genoma_crossover_final, GENLONG, best_partA, best_partB, best_partC, best_partD) ) 
            new_best    = worst_partA + best_partB + worst_partC + best_partD
            new_worst   = best_partA + worst_partB + best_partC + worst_partD

            endBest = new_best
            endWorst =  new_worst

            POPULATION_Y[j]["hyperparameters_chromosome"]     = new_best
            POPULATION_Y[N-j-1]["hyperparameters_chromosome"] = new_worst

            #print("\n\nCrossover Performed on individual {}-{}: \n\t StartBest: {} \n\t NewBest: {} \n\t StartWorst: {} \n\t NewWorst: {}".format(j,genoma_crossover, startBest, endBest, startWorst, endWorst))

    for j in range(N):
        #MUTATION

        genoma = POPULATION_Y[j]["hyperparameters_chromosome"]
        if mutation_type == "mutation_per_bit":
            mutated_genoma = _mutation_per_bit(genoma, PM)
        else:
            mutated_genoma = _mutation_per_individual(genoma, PM)

        POPULATION_Y[j]["hyperparameters_chromosome"] = mutated_genoma

    #print("\n\n0 ******************** deben ser iguales\nTEST POPULATION diff POPULATION_Y")
    #test_chromosome_xy(POPULATION_Y, POPULATION_X) #Prueba de que el genotipo ha sido efectivo
    #print("\n\n\n -------------------Here most mark Error")
    #test_fenotype_chromosome_POPULATION(POPULATION_Y)
    POPULATION_Y = change_fenotype_using_genoma(POPULATION_Y, default_param_dict)
    print("\n\n**Second test in test_fenotype_chromosome_POPULATION")
    test_fenotype_chromosome_POPULATION(POPULATION_Y)
    #print("\n\n2 ******************** deben cambiar \nTEST POPULATION diff POPULATION_Y")
    #test_chromosome_xy(POPULATION_Y, POPULATION_X) #Prueba de que el genotipo ha sido efectivo

    return POPULATION_Y



def  change_fenotype_using_genoma(POPULATION, default_param_dict):
    """ After modifying the genoma of an individual it's necessary to change the fenotype
    acordingly to the genotype. In other words this fucntion change the "shape" or characteristics
    of the individual (columns used in the model fitting)  ti fit the model.
    """
    for key in POPULATION.keys():
        hyperparameters_chromosome =  POPULATION[key]["hyperparameters_chromosome"]
        POPULATION[key]["model"]._update_hyperparameters(hyperparameters_chromosome, default_param_dict)
        POPULATION[key]["hyperparameters"] = POPULATION[key]["model"].hyperparameters
    test_fenotype_chromosome_POPULATION(POPULATION)
    return  POPULATION


def test_fenotype_chromosome_POPULATION(POPULATION):
    """ Test if the conversion of hyperparameters_chromosome is copied in the fitted model.+
    """
    print("Performing test on test_fenotype_chromosome_POPULATION: ")
    for k in POPULATION.keys():
        model = POPULATION[k]["model"]
        if model.hyperparameters != POPULATION[k]["hyperparameters"]:
            raise ValueError("Error: model.hyperparameters differ from POPULATION[k].hyperparameters")
        for param in model.hyperparameters_dictionary.keys():
            if model.hyperparameters_dictionary[param]["selected_value"] != POPULATION[k]["hyperparameters"][param]:
                raise ValueError("Error: model.hyperparameters_dictionary differ from model.hyperparameters and POPULATION[k].hyperparameters_chromosome")
    print("\t**TEST: test_fenotype_chromosome_POPULATION passed")

def test_fenotype_chromosome_SOLUTIONS(SOLUTIONS):
    print("Performing test on test_fenotype_chromosome_SOLUTIONS: ")
    for k in SOLUTIONS.keys():
        model = SOLUTIONS[k]["model"]
        if model.hyperparameters != SOLUTIONS[k]["hyperparameters"]:
            raise ValueError("Error: model.hyperparameters differ from SOLUTIONS[k].hyperparameters")
        for param in model.hyperparameters_dictionary.keys():
            if model.hyperparameters_dictionary[param]["selected_value"] != SOLUTIONS[k]["hyperparameters"][param]:
                raise ValueError("Error: model.hyperparameters_dictionary differ from model.hyperparameters and POPULATION[k].hyperparameters_chromosome")
    print("\t**TEST: test_fenotype_chromosome_SOLUTIONS passed")



def transform_KFold_random(df, y_column, K):

    df = df.reset_index()
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
    df_kfolded["all_data"]["index"] =  df.index
    df_kfolded["all_data"]["y"]     =  df_copy[y_column]
    del df_copy[y_column]
    df_kfolded["all_data"]["data"]  =  df_copy

    return df_kfolded


def transform_KFold_groups(df, y_column, K, group_id):
    """Generates a df_kfolded dictioanry. Each entry in the dictionary conatins a fold which have a 
    unique set of groups from the groups_id column that are not reapeated in any other fold.
    For example is group_id is the list od products from a store, then each fold will contain subset
    of these list and the interection of all the subsets will be null.
    Also the function have a balanced number of opservation in each fold """
    df = df.reset_index(drop = True)
    df_kfolded = dict()
    unique_vis = np.array(sorted(df[group_id].unique()))

    # Get folds
    folds = GroupKFold(n_splits=K)
    fold_ids= []
    ids = np.arange(df.shape[0])

    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df[group_id].isin(unique_vis[trn_vis])],
                ids[df[group_id].isin(unique_vis[val_vis])]
            ]
        )
    for k, (trn_, val_) in enumerate(fold_ids):
        df_kfolded[k] = dict()
        df_kfolded[k]["data"]  = df.iloc[val_]
        df_kfolded[k]["index"] = val_
        df_kfolded[k]["y"]     =  df.iloc[val_][y_column]
        del df_kfolded[k]["data"][y_column]

    df_copy = df.copy()
    df_kfolded["all_data"] = dict()
    df_kfolded["all_data"]["data"]  =  df_copy
    df_kfolded["all_data"]["index"] =  df.index
    df_kfolded["all_data"]["y"]     =  df_copy[y_column]
    del df_kfolded["all_data"]["data"][y_column]

    return df_kfolded

