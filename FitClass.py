from copy import copy, deepcopy
import torch
from torch import nn
import numpy as np


def weight_reset(layer):
    if type(layer) == nn.Linear:
        layer.reset_parameters()


def regularization(neural_network, norm_type):
    regularization_term = 0
    for parameter_name, parameter_value in neural_network.named_parameters():
        if any(term in parameter_name for term in ['weight', 'bias']):
            regularization_term += torch.norm(parameter_value, norm_type)
    return regularization_term


class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Ec, solution_network, test_network, initial_points, initial_values, 
               boundary_points, boundary_values, collocation_points, epoch, verbose=1, minimizing=True):
        residual_weight = solution_network.lambda_residual
        solution_reg_weight = solution_network.regularization_param
        test_reg_weight = test_network.regularization_param

        predicted_values = []
        target_values = []

        Ec.apply_bc(solution_network, boundary_points, boundary_values, predicted_values, target_values)
        if initial_points.size(0) > 0:
            Ec.apply_ic(solution_network, initial_points, initial_values, predicted_values, target_values)

        predicted_values_tensor = torch.cat(predicted_values, 0).to(Ec.device)
        target_values_tensor = torch.cat(target_values, 0).to(Ec.device)

        loss_values = torch.mean(torch.abs(predicted_values_tensor - target_values_tensor) ** Ec.p)

        solution_reg_loss = regularization(solution_network, 2)
        test_reg_loss = regularization(test_network, 2)

        if minimizing:
            if verbose:
                print("############### MINIMIZING ###############")
            pde_loss, pde_loss_no_norm = Ec.compute_res(solution_network, test_network, collocation_points, minimizing)
            total_loss = residual_weight * loss_values.to(Ec.device) + pde_loss.to(Ec.device) + solution_reg_weight * solution_reg_loss.to(Ec.device) + test_reg_weight * test_reg_loss.to(Ec.device)
            if verbose:
                print("###############################################################################################################")
                print("Function Loss    : ", (loss_values ** (1 / Ec.p)).detach().cpu().numpy(),
                      "\nPDE Residual     : ", (pde_loss_no_norm ** (1 / Ec.p)).detach().cpu().numpy())
                print()
                print()

            return total_loss, loss_values, pde_loss, pde_loss_no_norm
        else:
            if verbose:
                print("############### MAXIMIZING ###############")
            pde_loss, pde_loss_no_norm = Ec.compute_res(solution_network, test_network, collocation_points, minimizing)

            total_loss = - torch.log(pde_loss)
            return total_loss, 0, 0, [0, 0, 0], pde_loss_no_norm


def fit(Ec, solution_model, test_function_model, optimizer_min, optimizer_max, training_set_class, verbose= True):
    num_epochs = solution_model.num_epochs
    iterations_max = test_function_model.iterations
    iterations_min = solution_model.iterations
    reset_freq = int(solution_model.reset_freq * num_epochs)

    best_losses = list([0, 0, 0, 0, 0])
    freq = 1000

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal

    solution_model.train()
    test_function_model.train()

    if iterations_min != 0:
        best_train = 1e+12
    else:
        best_train = 0
    best_solution_model = None
    best_test_function_model = None

    lambda1 = lambda e: 1 / (1 + (e / num_epochs))
    my_lr_scheduler_min = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_min, lr_lambda=lambda1)
    my_lr_scheduler_max = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_max, lr_lambda=lambda1)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        if epoch % reset_freq == 0 and epoch != 0:
            print("Resetting Params")
            test_function_model.apply(weight_reset)

        current_losses = list([0, 0, 0, 0, 0])

        def closure_max():
            optimizer_max.zero_grad()
            loss_test, _, _, _, res_pde_no_norm \
                = CustomLoss().forward(Ec, solution_model, test_function_model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, epoch, verbose, False)
            current_losses[4] = current_losses[4] + float(res_pde_no_norm.cpu().detach().numpy())
            loss_test.backward()
            return loss_test

        def closure_min():
            optimizer_min.zero_grad()
            loss_sol, loss_vars, loss_int, res_pde_no_norm = CustomLoss().forward(Ec, solution_model, test_function_model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, epoch, verbose, True)
            current_losses[0] = current_losses[0] + float(loss_sol.cpu().detach().numpy())
            current_losses[1] = current_losses[1] + float(loss_vars.cpu().detach().numpy())
            current_losses[2] = current_losses[2] + float(loss_int.cpu().detach().numpy())
            current_losses[3] = current_losses[3] + float(res_pde_no_norm.cpu().detach().numpy())

            loss_sol.backward()
            return loss_sol

        if epoch % freq == 0:
            print("##################################################  ", epoch, "  ##################################################")
            L2_test, rel_L2_test = Ec.compute_generalization_error(solution_model)
            print(f"Epoch {epoch}: L2 Test Error: {L2_test}, Relative L2 Error: {rel_L2_test}")

        batch = 0
        if len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):

                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                for _ in range(iterations_max):
                    optimizer_max.step(closure=closure_max)

                for _ in range(iterations_min):
                    optimizer_min.step(closure=closure_min)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()
                batch = batch + 1

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_initial_internal, training_boundary)):
                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                for _ in range(iterations_max):
                    optimizer_max.step(closure=closure_max)

                for _ in range(iterations_min):
                    optimizer_min.step(closure=closure_min)

                if torch.cuda.is_available():
                    del x_coll_train_
                    del x_b_train_
                    del u_b_train_
                    del x_u_train_
                    del u_train_
                    torch.cuda.empty_cache()

                batch = batch + 1
        for l in range(len(current_losses)):
            current_losses[l] = current_losses[l] / batch

        if np.isnan(current_losses[0]):
            print("WARNING: Found NaN")
            return best_losses, best_solution_model, best_test_function_model

        if current_losses[0] < best_train:
            best_solution_model = deepcopy(solution_model)
            best_test_function_model = deepcopy(test_function_model)
            best_losses[0] = current_losses[0]
            best_losses[1] = current_losses[1]
            best_losses[2] = current_losses[2]
            best_losses[3] = current_losses[3]
            best_losses[4] = current_losses[4]
            best_train = current_losses[0]

        my_lr_scheduler_min.step()
        my_lr_scheduler_max.step()
    return best_losses, best_solution_model, best_test_function_model



