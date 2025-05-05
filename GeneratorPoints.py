import torch
import numpy as np
import sobol_seq


def generator_points(samples, dim, random_seed, type_of_points, boundary):
    if type_of_points == "random":
        torch.random.manual_seed(random_seed)
        return torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_of_points == "gauss":
        if samples > 0:
            nodes, _ = np.polynomial.legendre.leggauss(samples)
            normalized_nodes = 0.5 * (nodes.reshape(-1, 1) + 1)
            
            if dim == 1:
                return torch.from_numpy(normalized_nodes).type(torch.FloatTensor)
            if dim == 2:
                flattened = normalized_nodes.flatten()
                grid_points = np.transpose([np.repeat(flattened, len(flattened)), 
                                           np.tile(flattened, len(flattened))])
                return torch.from_numpy(grid_points).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])
    elif type_of_points == "grid":
        if samples > 0:
            points_per_axis = int(samples ** (1 / dim))
            
            if dim == 1:
                grid = np.linspace(0, 1, points_per_axis + 2)
                interior_points = grid[1:-1]
                return torch.from_numpy(interior_points).type(torch.FloatTensor)
            if dim == 2:
                if not boundary:
                    grid = np.linspace(0, 1, points_per_axis + 2)
                    interior_points = grid[1:-1]
                    y_coords = np.copy(interior_points)
                    inputs = np.transpose([np.tile(interior_points, len(y_coords)), 
                                           np.repeat(y_coords, len(interior_points))])
                else:
                    grid = np.linspace(0, 1, samples)
                    inputs = np.concatenate([grid.reshape(-1, 1), grid.reshape(-1, 1)], 1)
                print(inputs)
                print(inputs.shape)
                return torch.from_numpy(inputs).type(torch.FloatTensor)
        else:
            return torch.zeros([0, dim])

    elif type_of_points == "sobol":
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        return torch.from_numpy(data).type(torch.FloatTensor)


