import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader


class DefineDataset:
    def __init__(self, 
                 Ec,
                 n_collocation,
                 n_boundary,
                 n_initial,
                 n_internal,
                 batches,
                 random_seed,
                 shuffle=False):
        
        # Store equation configuration and parameters
        self.equation_config = Ec
        self.points_collocation = n_collocation
        self.points_boundary = n_boundary
        self.points_initial = n_initial
        self.points_internal = n_internal
        self.batch_size = "full" if batches == "full" else int(batches)
        self.seed = random_seed
        self.enable_shuffle = shuffle
        
        # Extract dimensions from equation configuration
        self.space_dims = self.equation_config.space_dimensions
        self.time_dims = self.equation_config.time_dimensions
        self.input_dims = self.space_dims + self.time_dims
        self.output_dims = self.equation_config.output_dimension
        
        # Calculate total sample count
        self.total_samples = (self.points_collocation + 
                             2 * self.points_boundary * self.space_dims + 
                             self.points_initial * self.time_dims + 
                             self.points_internal)
        
        # Initialize data containers
        self.BC = None
        self.data_coll = None
        self.data_boundary = None
        self.data_initial_internal = None

    def assemble_dataset(self):

        fraction_coll = int(self.batch_size * self.points_collocation / self.total_samples)
        fraction_boundary = int(self.batch_size * 2 * self.points_boundary * self.space_dims / self.total_samples)
        fraction_initial = int(self.batch_size * self.points_initial / self.total_samples)
        fraction_internal = int(self.batch_size * self.points_internal / self.total_samples)

        x_coll, y_coll = self.equation_config.add_collocation_points(self.points_collocation, self.seed)
        x_b, y_b = self.equation_config.add_boundary_points(self.points_boundary, self.seed)

        if self.points_initial == 0:
            x_time_internal = torch.zeros((self.points_initial, self.input_dims))
            y_time_internal = torch.zeros((self.points_initial, self.output_dims))
        else:
            x_time_internal, y_time_internal = self.equation_config.add_initial_points(self.points_initial, self.seed)

        if self.points_internal != 0:
            x_internal, y_internal = self.equation_config.add_internal_points(self.points_internal, self.seed)
            x_time_internal = torch.cat([x_time_internal, x_internal])
            y_time_internal = torch.cat([y_time_internal, y_internal])

        if self.points_collocation == 0:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=1, shuffle=False)
        else:
            self.data_coll = DataLoader(torch.utils.data.TensorDataset(x_coll, y_coll), batch_size=fraction_coll, shuffle=self.enable_shuffle)

        if self.points_boundary == 0:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=1, shuffle=False)
        else:
            self.data_boundary = DataLoader(torch.utils.data.TensorDataset(x_b, y_b), batch_size=fraction_boundary, shuffle=self.enable_shuffle)

        if fraction_internal == 0 and fraction_initial == 0:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal), batch_size=1, shuffle=False)
        else:
            self.data_initial_internal = DataLoader(torch.utils.data.TensorDataset(x_time_internal, y_time_internal), batch_size=fraction_initial + fraction_internal,
                                                    shuffle=self.enable_shuffle)
