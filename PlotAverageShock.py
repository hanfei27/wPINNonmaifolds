from ShockRarEntropy import EquationClass
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd


def calculate_weight_function(input_tensor):
    dimension_count = input_tensor.shape[1]
    normalization_constant = 1
    transformed_input = torch.zeros_like(input_tensor)
    weight_components = torch.zeros_like(input_tensor)
    
    for dimension in range(1, dimension_count):
        upper_bound = 1
        lower_bound = -1
        half_range = (upper_bound - lower_bound) / 2.0
        transformed_input[:, dimension] = (input_tensor[:, dimension] - lower_bound - half_range) / half_range
    
    for dimension in range(1, dimension_count):
        support_condition = torch.gt(torch.tensor(1.0) - torch.abs(transformed_input[:, dimension]), 0)
        exponential_term = torch.exp(torch.tensor(1.0) / (transformed_input[:, dimension] ** 2 - 1)) / normalization_constant
        weight_components[:, dimension] = torch.where(support_condition, exponential_term, torch.zeros_like(transformed_input[:, dimension]))
    
    final_weight = torch.ones_like(input_tensor[:, 0])
    for dimension in range(1, dimension_count):
        final_weight = final_weight * weight_components[:, dimension]
    
    return final_weight / np.max(final_weight.cpu().detach().numpy())


output_directories = ["moving_e5000_re10"]
description_suffix = "rar"
equation_instance = EquationClass(None, None, None, None)

use_gaussian = False
use_sine = False
N_s = 10

experiment_number = 1
enable_plotting = True
test_entropy = False
final_time_step = False

if not use_gaussian:
    exact_solution_file = "Data/BurgersExact.txt"
    exact_solution = np.loadtxt(exact_solution_file)
    input_data = torch.from_numpy(exact_solution[np.where((np.isclose(exact_solution[:, 0], 0, 1e-3)) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2])
    input_data = input_data.reshape(input_data.shape[1], 2)
else:
    exact_solution_file = "Data/DataGauss.txt"
    exact_solution = np.loadtxt(exact_solution_file)
    input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], 0.75, 2e-3)), :-1]).type(torch.float32)
    input_data = input_data.reshape(input_data.shape[1], 2)
if use_sine:
    time_steps = [0.0, 0.25, 0.5, 0.75]
    time = 1
if use_gaussian:
    time_steps = [0.0018, 0.25, 0.5, 0.75]
    time = 0.75
if not use_sine and not use_gaussian:
    time_steps = [0.0, 0.25, 0.45]
    time = 0.4
scale_vector = np.linspace(0.65, 1.55, len(time_steps))

N_s_list = list([N_s])
best_error = list()
average_error = list()

for N_s in N_s_list:
    solution_u = torch.zeros((N_s, len(time_steps), input_data.shape[0]))
    solution_test = torch.zeros((N_s, len(time_steps), input_data.shape[0]))
    solution_u_ex = torch.zeros((N_s, len(time_steps), input_data.shape[0]))
    average_solution = 0
    best_solution = 0
    best_model_index_train = None
    smallest_train_error = 10

    if test_entropy:
        average_test_entropy = 0
        solution_test_entropy = torch.zeros((N_s, len(time_steps), input_data.shape[0]))
    for output_directory in output_directories:
        if not use_gaussian:
            if final_time_step:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(
                    torch.FloatTensor)
            else:
                input_data = torch.from_numpy(exact_solution[np.where((exact_solution[:, 0] < time) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(
                    torch.FloatTensor)
        else:
            if final_time_step:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 2e-3)), :-1]).type(torch.float32)
            else:
                input_data = torch.from_numpy(exact_solution[np.where(exact_solution[:, 0] < time), :-1]).type(torch.float32)
        input_data = input_data.reshape(input_data.shape[1], 2)
        print("#################################################")

        model_directories = [d for d in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, d))]
        print(model_directories)

        for i, retrain_path in enumerate(model_directories):
            if os.path.isfile(output_directory + "/" + retrain_path + "/InfoModel.txt"):
                print(output_directory + "/" + retrain_path + "/InfoModel.txt ")
                model_info = pd.read_csv(output_directory + "/" + retrain_path + "/InfoModel.txt", header=None, sep=",", index_col=0)
                model_info = model_info.transpose().reset_index().drop("index", 1)
                model_info["metric"] = model_info["loss_pde_no_norm"] + model_info["loss_vars"]
                pickle_inn_sol = open(output_directory + "/" + retrain_path + "/ModelSol.pkl", 'rb')

                model_solution = torch.load(pickle_inn_sol)

                solution_outputs = model_solution(input_data)[:, 0].detach()
                if test_entropy:
                    pickle_inn_test_entropy = open(output_directory + "/" + retrain_path + "/ModelEntropy.pkl", 'rb')
                    model_test_entropy = torch.load(pickle_inn_test_entropy)

                    test_entropy_outputs = (model_test_entropy(input_data)[:, 0].detach()) ** experiment_number * calculate_weight_function(input_data)

                if i < N_s:
                    average_solution = average_solution + solution_outputs / N_s

                    print("##############################")
                    print(retrain_path)
                    print(model_info["rel_L2_norm"].values)
                    print(model_info["metric"].values)

                    if model_info["metric"].values < smallest_train_error:
                        best_model_index_train = i
                        best_solution = solution_outputs
                        smallest_train_error = model_info["metric"].values

                for k, val in enumerate(time_steps):
                    if not use_gaussian:

                        input_data_val = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
                    else:
                        input_data_val = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
                    input_data_val = input_data_val.reshape(input_data_val.shape[1], 2)
                    solution_u_outputs = model_solution(input_data_val)[:, 0].detach()
                    solution_u[i, k, :] = solution_u_outputs
            else:
                print(output_directory + "/" + retrain_path + "/InfoModel.txt not found")

    best_train_model_u = solution_u[best_model_index_train, :, :]
    best_train_model_test = solution_test[best_model_index_train, :, :]

    mean_u = torch.mean(solution_u, 0)
    std_u = torch.std(solution_u, 0)

    if not use_sine and not use_gaussian:
        ex = equation_instance.exact(input_data).detach().numpy()
    if use_sine:
        if final_time_step:
            ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1]
        else:
            ex = exact_solution[np.where((exact_solution[:, 0] < time) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1]

    if use_gaussian:
        if final_time_step:
            ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], time, 2e-3)), -1]
        else:
            ex = exact_solution[np.where(exact_solution[:, 0] < time), -1]
        ex = ex.reshape(-1, )

    average_l1_error = np.mean(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, ))) / np.mean(abs(ex.reshape(-1, )))
    average_l2_error = (np.mean(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, )) ** 2) / np.mean(abs(ex.reshape(-1, )) ** 2)) ** 0.5
    average_max_error = np.max(abs(ex.reshape(-1, ) - average_solution.detach().numpy().reshape(-1, ))) / np.max(abs(ex.reshape(-1, )))

    best_l1_error = np.mean(abs(ex.reshape(-1, ) - best_solution.detach().numpy()).reshape(-1, )) / np.mean(abs(ex.reshape(-1, )))
    best_l2_error = (np.mean(abs(ex.reshape(-1, ) - best_solution.detach().numpy().reshape(-1, )) ** 2) / np.mean(abs(ex.reshape(-1, )) ** 2)) ** 0.5
    print("Average L1 error:", average_l1_error)
    print("Average L2 error:", average_l2_error)
    print("Average Linf error:", average_max_error)
    print("Best Trained L1 error:", best_l1_error)
    print("Best Trained L2 error:", best_l2_error)
    average_error.append(average_l1_error)
    if enable_plotting:

        p = 1

        fig = plt.figure()
        plt.grid(True, which="both", ls=":")
        for k, (val, scale) in enumerate(zip(time_steps, scale_vector)):
            if not use_gaussian:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
            else:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
            input_data = input_data.reshape(input_data.shape[1], 2)
            x_plot = input_data[:, 1].reshape(-1, 1)

            if not use_sine and not use_gaussian:
                ex = equation_instance.exact(input_data)
            if use_sine:
                ex = exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1].reshape(-1, 1)
            if use_gaussian:
                ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), -1]
                ex = ex.reshape(-1, )

            plt.plot(x_plot.cpu().detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=equation_instance.lighten_color('grey', scale), zorder=0)
            plt.plot(x_plot.cpu().detach().numpy(), mean_u[k, :], label=r'Predicted, $t=$' + str(val) + r'$s$', color=equation_instance.lighten_color('C0', scale), zorder=10)
            plt.fill_between(x_plot.cpu().detach().numpy().reshape(-1, ), mean_u[k, :] - 2 * std_u[k, :], mean_u[k, :] + 2 * std_u[k, :], alpha=0.25, color="grey")

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(output_directories[0] + "/u_average_" + description_suffix + ".png", dpi=500)

        fig = plt.figure()
        plt.grid(True, which="both", ls=":")
        for k, (val, scale) in enumerate(zip(time_steps, scale_vector)):

            if not use_gaussian:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 1e-3) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), :2]).type(torch.FloatTensor)
            else:
                input_data = torch.from_numpy(exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), :-1]).type(torch.float32)
            input_data = input_data.reshape(input_data.shape[1], 2)
            x_plot = input_data[:, 1].reshape(-1, 1)

            if not use_sine and not use_gaussian:
                ex = equation_instance.exact(input_data)
            if use_sine:
                ex = exact_solution[np.where((exact_solution[:, 0] == val) & (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == 0.0)), -1].reshape(-1, 1)
            if use_gaussian:
                ex = exact_solution[np.where(np.isclose(exact_solution[:, 0], val, 2e-3)), -1]
                ex = ex.reshape(-1, )

            plt.plot(x_plot.cpu().detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=equation_instance.lighten_color('grey', scale), zorder=0)
            plt.plot(x_plot.cpu().detach().numpy(), best_train_model_u[k, :], label=r'Predicted, $t=$' + str(val) + r'$s$', color=equation_instance.lighten_color('C0', scale), zorder=10)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.legend()
        plt.savefig(output_directories[0] + "/u_best_trained_" + description_suffix + ".png", dpi=500)
