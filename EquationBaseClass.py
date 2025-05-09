from torch import device, cuda, mean, abs
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as mc
import colorsys
from scipy.special import legendre
import sobol_seq
import torch


class EquationBaseClass:
    def __init__(self, norm, cutoff, weak_form, p=2):
        self.normalization_type = norm
        self.cutoff_method = cutoff
        self.weak_formulation = weak_form
        self.computation_device = device('cuda') if cuda.is_available() else device("cpu")
        self.power_parameter = p
        
    def return_norm(self, func, func_x, func_y):
        power = self.power_parameter
        
        if self.normalization_type == "H1":
            combined_terms = abs(func)**power + abs(func_x)**power + abs(func_y)**power
            result = mean(combined_terms)**(1/power)
        elif self.normalization_type == "L2":
            result = mean(abs(func)**power)**(1/power)
        elif self.normalization_type == "H1s":
            derivative_terms = abs(func_x)**power + abs(func_y)**power
            result = mean(derivative_terms)**(1/power)
        elif self.normalization_type is None:
            result = 1
        else:
            raise ValueError("Unsupported normalization type")
            
    # Weak Adversarial Networks for High-dimensional Partial
    # Differential Equations,
    # Yaohua Zang, Gang Bao, Xiaojing Ye, Haomin Zhou

    def fun_w(self, x, extrema_values, time_dimensions):
        if self.cutoff_method == "def_max":
            dim = x.shape[1]
            I1 = 1
            x_mod = torch.zeros_like(x)
            x_mod_2 = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            for i in range(time_dimensions, dim):
                supp_x = torch.gt(torch.tensor(1.0) - torch.abs(x_mod[:, i]), 0)
                x_mod_2[:, i] = torch.where(supp_x, torch.exp(torch.tensor(1.0) / (x_mod[:, i] ** 2 - 1)) / I1, torch.zeros_like(x_mod[:, i]))
            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * x_mod_2[:, i]

            return w / np.max(w.cpu().detach().numpy())
        if self.cutoff_method == "def_av":
            x.requires_grad = True
            dim = x.shape[1]
            I1 = 0.210987

            x_mod = torch.zeros_like(x)
            x_mod_2 = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            for i in range(time_dimensions, dim):
                supp_x = torch.gt(torch.tensor(1.0) - torch.abs(x_mod[:, i]), 0)
                x_mod_2[:, i] = torch.where(supp_x, torch.exp(torch.tensor(1.0) / (x_mod[:, i] ** 2 - 1)) / I1, torch.zeros_like(x_mod[:, i]))
            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * x_mod_2[:, i]
            return w
        elif self.cutoff_method == "net":
            w = torch.load("EnsDist/Setup_25/Retrain_4/ModelInfo.pkl")
            for param in w.parameters():
                param.requires_grad = False
            x = (x - extrema_values[:, 0]) / (extrema_values[:, 1] - extrema_values[:, 0])
            return w(x)
        if self.cutoff_method == "quad":
            dim = x.shape[1]
            x_mod = torch.zeros_like(x)

            for i in range(time_dimensions, dim):
                up = extrema_values[i, 1]
                low = extrema_values[i, 0]
                h_len = (up - low) / 2.0
                x_mod[:, i] = (x[:, i] - low - h_len) / h_len

            w = 1.0
            for i in range(time_dimensions, dim):
                w = w * (-x_mod[:, i] ** 2 + 1)

            return w / np.max(w.cpu().detach().numpy())
        else:
            raise ValueError

    def convert(self, vector, extrema_values):
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        vector = np.array(vector)

        max_val = np.max(np.array(extrema_values), axis=1)
        min_val = np.min(np.array(extrema_values), axis=1)

        vector = vector * (max_val - min_val) + min_val

        return torch.from_numpy(vector).type(torch.FloatTensor)

    def lighten_color(self, color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
