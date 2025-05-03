import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def select_over_retrainings(folder_path, selection="error_train", mode="min", exact_solution=None):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]
        if os.path.isfile(retrain_path + "/InfoModel.txt"):

            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=None, sep=",", index_col=0)
            models = models.transpose()
            models["metric"] = models["loss_pde_no_norm"] + models["loss_vars"]
            models["retraining"] = number_of_ret

            models_list.append(models)
        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    retraining_prop = retraining_prop.sort_values(selection)
    if mode == "min":
        return retraining_prop.iloc[0]
    if mode == "max":
        return retraining_prop.iloc[-1]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop


def select_over_retrainings_dist(folder_path, selection="error_train", mode="min"):
    subdirectories = [directory for directory in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, directory))]
    
    collected_models = []
    for directory in subdirectories:
        full_path = os.path.join(folder_path, directory)
        info_file_path = os.path.join(full_path, "InfoModel.txt")
        training_id = directory.split("_")[-1]
        
        if os.path.exists(info_file_path):
            model_data = pd.read_csv(info_file_path, header=None, sep=",", index_col=0)
            model_data = model_data.transpose()
            model_data["retraining"] = training_id
            collected_models.append(model_data)
        else:
            print("No File Found")
    
    all_results = pd.concat(collected_models, ignore_index=True)
    sorted_results = all_results.sort_values(selection)
    
    if mode == "min":
        return sorted_results.iloc[0]
    elif mode == "max":
        return sorted_results.iloc[-1]
    else:
        selected_training = sorted_results["retraining"].iloc[0]
        average_results = sorted_results.mean()
        average_results["retraining"] = selected_training
        return average_results
