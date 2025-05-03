from os import path, mkdir, system
from sys import argv, platform
from itertools import product

random_seed = 32
collocation_points = int(argv[1])
boundary_points = int(argv[2])
internal_points = int(argv[3])
output_directory = argv[4]
validation_proportion = 0.0

model_config = {
    "hidden_layers_sol": [4, 6],
    "hidden_layers_test": [2, 4],
    "neurons_sol": [20],
    "neurons_test": [10],
    "activation_sol": ["tanh"],
    "activation_test": ["tanh", "sin"],
    "tau_sol": [0.01],
    "tau_test": [0.015],
    "iterations_min": [1],
    "iterations_max": [6, 8],
    "residual_parameter": [10],
    "kernel_regularizer": [2],
    "regularization_parameter_sol": [0],
    "regularization_parameter_test": [0],
    "batch_size": [collocation_points + boundary_points + internal_points],
    "epochs": [5000],
    "norm": ["H1"],
    "cutoff": ["def_max"],
    "weak_form": ["partial"],
    "reset_freq": [0.025, 0.05, 0.25],
    "loss_type": ["l2"],
}

shuffle = "false"
cluster = argv[5]
GPU = "None"
n_retrain = 10

if not path.isdir(output_directory):
    mkdir(output_directory)
settings = list(product(*model_config.values()))

i = 0
for setup in settings:
    print(setup)

    folder_path = output_directory + "/Setup_" + str(i)
    print("###################################")
    setup_properties = {
        "hidden_layers_sol": setup[0],
        "hidden_layers_test": setup[1],
        "neurons_sol": setup[2],
        "neurons_test": setup[3],
        "activation_sol": setup[4],
        "activation_test": setup[5],
        "tau_sol": setup[6],
        "tau_test": setup[7],
        "iterations_min": setup[8],
        "iterations_max": setup[9],
        "residual_parameter": setup[10],
        "kernel_regularizer": setup[11],
        "regularization_parameter_sol": setup[12],
        "regularization_parameter_test": setup[13],
        "batch_size": setup[14],
        "epochs": setup[15],
        "norm": setup[16],
        "cutoff": setup[17],
        "weak_form": setup[18],
        "reset_freq": setup[19],
        "loss_type": setup[20]
    }

    arguments = list()
    arguments.append(str(random_seed))
    arguments.append(str(collocation_points))
    arguments.append(str(boundary_points))
    arguments.append(str(internal_points))
    arguments.append(str(folder_path))
    arguments.append(str(validation_proportion))
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))
    arguments.append(str(GPU))
    arguments.append(str(n_retrain))

    if platform == "linux" or platform == "linux2" or platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 SingleRetraining.py "
        else:
            string_to_exec = "python3 SingleRetraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        system(string_to_exec)
    i = i + 1
