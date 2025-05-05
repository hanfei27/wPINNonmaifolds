import torch
import numpy as np
from GeneratorPoints import generator_points


class SquareDomain:
    def __init__(self, output_dimension, time_dimensions, space_dimensions, list_of_BC, extrema_values, type_of_points):
        self.domain_output_size = output_dimension
        self.temporal_dimensions = time_dimensions
        self.spatial_dimensions = space_dimensions
        self.boundary_conditions = list_of_BC
        self.domain_bounds = extrema_values
        self.sampling_strategy = type_of_points

        self.total_dimensions = self.temporal_dimensions + self.spatial_dimensions
        self.lower_bounds = self.domain_bounds[:, 0]
        self.upper_bounds = self.domain_bounds[:, 1]

        self.boundary_types = list()

    def add_collocation_points(self, n_coll, random_seed):
        interior_points = generator_points(n_coll, self.total_dimensions, random_seed, self.sampling_strategy, False)
        scaled_points = interior_points * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        collocation_values = torch.full((scaled_points.shape[0], self.domain_output_size), np.nan)
        return scaled_points, collocation_values

    def add_boundary_points(self, n_boundary, random_seed):
        boundary_coordinates = list()
        boundary_values = list()

        for dimension in range(self.temporal_dimensions, self.temporal_dimensions + self.spatial_dimensions):
            boundary_condition_pair = list()
            remaining_min_values = np.delete(self.domain_bounds, dimension, 0)[:, 0]
            remaining_max_values = np.delete(self.domain_bounds, dimension, 0)[:, 1]
            left_boundary = generator_points(n_boundary, self.total_dimensions, random_seed, self.sampling_strategy, True)

            left_boundary[:, dimension] = torch.full(size=(n_boundary,), fill_value=0.0)
            left_boundary_wo_dim = np.delete(left_boundary, dimension, 1)

            [left_boundary_values, boundary_type] = self.boundary_conditions[dimension - self.temporal_dimensions][0](left_boundary_wo_dim * (remaining_max_values - remaining_min_values) + remaining_min_values)

            boundary_condition_pair.append(boundary_type)
            boundary_coordinates.append(left_boundary)
            boundary_values.append(left_boundary_values)
            right_boundary = generator_points(n_boundary, self.total_dimensions, random_seed, self.sampling_strategy, True)
            right_boundary[:, dimension] = torch.tensor(()).new_full(size=(n_boundary,), fill_value=1.0)
            right_boundary_wo_dim = np.delete(right_boundary, dimension, 1)

            [right_boundary_values, boundary_type] = self.boundary_conditions[dimension - self.temporal_dimensions][1](
                right_boundary_wo_dim * (remaining_max_values - remaining_min_values) + remaining_min_values)
            boundary_condition_pair.append(boundary_type)

            self.boundary_types.append(boundary_condition_pair)

            boundary_coordinates.append(right_boundary)
            boundary_values.append(right_boundary_values)

        boundary_points = torch.cat(boundary_coordinates, 0)
        boundary_values = torch.cat(boundary_values, 0)

        boundary_points = boundary_points * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        return boundary_points, boundary_values

    def apply_boundary_conditions(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        for j in range(self.domain_output_size):
            for i in range(self.spatial_dimensions):
                half_len_x_b_train_i = int(x_b_train.shape[0] / (2 * self.spatial_dimensions))

                x_b_train_i = x_b_train[i * int(x_b_train.shape[0] / self.spatial_dimensions):(i + 1) * int(x_b_train.shape[0] / self.spatial_dimensions), :]
                u_b_train_i = u_b_train[i * int(x_b_train.shape[0] / self.spatial_dimensions):(i + 1) * int(x_b_train.shape[0] / self.spatial_dimensions), :]

                boundary = 0
                while boundary < 2:
                    x_b_train_i_half = x_b_train_i[half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]
                    u_b_train_i_half = u_b_train_i[half_len_x_b_train_i * boundary:half_len_x_b_train_i * (boundary + 1), :]

                    boundary_conditions = self.boundary_types[i][boundary][j]

                    u_b_pred, u_b_train_reshaped = boundary_conditions.apply(model, x_b_train_i_half, u_b_train_i_half, j)
                    u_pred_var_list.append(u_b_pred)
                    u_train_var_list.append(u_b_train_reshaped)

                    boundary = boundary + 1

