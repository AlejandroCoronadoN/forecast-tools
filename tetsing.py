
default_param_dict = {"eval_metric": "rmse",
                      "seed": 100,
                      "objective":  "reg:linear",
                      "nthread": 1,
                      "silent": False,
                      }

def generate_model_population(N, model, default_param_dict):
    POPULATION = dict()
    params_dictionary = model.params_dictionary
    hyperparameters_chromosome_size = model.hyperparameters_chromosome_size

    for i in range(N):
        model_i = copy.deepcopy(model)
        POPULATION[i] = dict()
        baseline_features_selected = list()
        hyperparameters_chromosome = list("0"*hyperparameters_chromosome_size)

        for i in  range(len(hyperparameters_chromosome)):
            is1 = random.choice([True, False])
            if is1:
                hyperparameters_chromosome[i] = "1"
        hyperparameters_chromosome = "".join( hyperparameters_chromosome)

        model_i._update_params(hyperparameters_chromosome)
        model_i._set_default_params(default_param_dict)

        params = model_i.params

        POPULATION[i]["model"] =  model_i
        POPULATION[i]["hyperparameters"] = params
        POPULATION[i]["hyperparameters_chromosome"] = hyperparameters_chromosome
        POPULATION[i]["GENOMA"] = POPULATION[i]["hyperparameters_chromosome"] #+""+ POPULATION[i]["exmodel_features_chromosome"]
        POPULATION[i]["SCORE"]  = np.nan

    return POPULATION
