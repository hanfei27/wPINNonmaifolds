import numpy as np
import torch
import matplotlib.pyplot as plt
from EquationBaseClass import EquationBaseClass
from SquareDomain import SquareDomain
from BoundaryConditions import DirichletBC
from GeneratorPoints import generator_points
from scipy.interpolate import NearestNDInterpolator, interp2d

torch.cuda.set_device(0)
print(f"Current GPU device index: {torch.cuda.current_device()}")
pi = np.pi

class EquationClass(EquationBaseClass):

    def __init__(self, norm, cutoff, weak_form, p):
        super().__init__(norm, cutoff, weak_form, p)

        self.sampling_method = "random"
        self.output_dimension = 1
        self.equation_type = "standing"
        
        if self.equation_type == "sine":
            self.domain_boundaries = torch.tensor([
                [0, 1],       # time domain: [0,T]
                [-1., 1.],    # first spatial domain 
                [-1., 1.]     # second spatial domain
            ])
            self.adjusted_boundaries = torch.tensor([
                [0, 1],
                [-0.997, 0.997],
                [-1., 1.]
            ])
        elif self.equation_type == "gauss":
            self.domain_boundaries = torch.tensor([
                [0, 0.75],    # time domain
                [-1., 1.]     # spatial domain
            ])
        else:
            self.domain_boundaries = torch.tensor([
                [0, 0.45],    # time domain
                [-1., 1.],    # first spatial domain
                [-0.5, 0.5]   # second spatial domain
            ])
            
        self.parameters_values = None
        self.time_dimensions = 1
        self.space_dimensions = 2
        self.parameter_dimensions = self.parameters_values.shape[0] if self.parameters_values is not None else 0
        self.list_of_BC = [[self.ubx0, self.ubx1],
                           [self.uby0, self.uby0]]
        self.nu = 0. / pi

        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.domain_boundaries,
                                          self.sampling_method)

        self.c_mode = "max"
        self.use_relu = True

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_internal_points(self, n_internal, random_seed):
        if self.equation_type != "gauss":
            x_int = generator_points(n_internal, self.time_dimensions + self.space_dimensions, random_seed,
                                     self.sampling_method, True)

            y_int = self.exact(x_int)
        else:
            np.random.seed(random_seed)
            file_ex = "Data/DataGauss.txt"
            exact_solution = np.loadtxt(file_ex)
            u = torch.from_numpy(exact_solution[:, -1]).type(torch.float32)
            inputs = torch.from_numpy(exact_solution[:, :-1]).type(torch.float32)

            idx = np.random.randint(0, inputs.shape[0], n_internal)
            x_int = inputs[idx]
            y_int = u[idx].reshape(-1, 1)
        return x_int, y_int

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.domain_boundaries[:, 0]
        extrema_f = self.domain_boundaries[:, 1]
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions, random_seed,
                                    self.sampling_method, True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        val_0 = self.domain_boundaries[1:, 0]
        val_f = self.domain_boundaries[1:, 1]

        x_time_wo_i = x_time_0[:, 1:]

        y_time_0 = self.u0(x_time_wo_i * (val_f - val_0) + val_0)

        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        return x_time_0, y_time_0

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u_pred_var_list.append(model(x_u_train)[:, j])
                u_train_var_list.append(u_train[:, j])

    def sign(self, x):
        mu = 0.01
        return 2 * torch.sigmoid(x / mu) - 1

    def abs(self, x):
        mu = 0.01
        return 2 * mu * torch.log(0.5 * (1 + torch.exp(x / mu))) - x

    def compute_res(self, network_sol, network_test, x_f_train, minimizing):
        device = x_f_train.device 

        x_f_train.requires_grad = True

        u = network_sol(x_f_train).reshape(-1, )
        theta = network_test(x_f_train).reshape(-1, )
        w = self.fun_w(x_f_train, self.domain_boundaries, self.time_dimensions).reshape(-1, )
        phi = theta ** 2

        min_c = -2
        max_c = 2

        N = 401
        c_vec = torch.linspace(float(min_c), float(max_c), N)

        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_t = grad_u[:, 0].reshape(-1, )

        grad_phi_x = torch.autograd.grad(phi, x_f_train, grad_outputs=torch.ones_like(phi), create_graph=True)[0][:,
                     1].reshape(-1, )
        grad_phi_y = torch.autograd.grad(phi, x_f_train, grad_outputs=torch.ones_like(phi), create_graph=True)[0][:,
                     2].reshape(-1, )

        norm_test = self.return_norm(phi, grad_phi_x, grad_phi_y) ** 2


        res_pde_no_norm_mean = torch.tensor(0., device=device)
        res_pde_no_norm_max = torch.tensor(0., device=device)

        for i, c in enumerate(c_vec):

            x2 = x_f_train[:, 2]  
            cos_x2 = torch.cos(0.5 * pi * x2)
            if self.weak_form == "partial":
                res_pde_no_norm_i = torch.mean(
                    self.sign(u - c) * (grad_u_t * phi - grad_phi_x * (u * u / 2.0 - c * c / 2.0)) * cos_x2)
                if self.use_relu:
                    res_pde_no_norm_i = torch.relu(res_pde_no_norm_i) + 1e-12

                res_pde_no_norm_i = network_sol.loss(torch.tensor(0., device=res_pde_no_norm_i.device),
                                                     res_pde_no_norm_i)
            else:
                res_pde_no_norm_i = torch.relu(-torch.mean(
                    self.sign(u - c) * (grad_u_t * phi - grad_phi_x * (u * u / 2.0 - c * c / 2.0)) * cos_x2)) ** 2

            res_pde_no_norm_mean = res_pde_no_norm_i / N + res_pde_no_norm_mean
            res_pde_no_norm_max = max(res_pde_no_norm_i, res_pde_no_norm_max)

        if self.c_mode == "max":
            res_pde_no_norm = res_pde_no_norm_max
        elif self.c_mode == "mean":
            res_pde_no_norm = res_pde_no_norm_mean
        elif self.c_mode == "both":
            res_pde_no_norm = res_pde_no_norm_mean + res_pde_no_norm_max
        else:
            raise ValueError()

        res_pde = 100 * res_pde_no_norm / norm_test

        return res_pde, res_pde_no_norm

    def compute_res_sol(self, network_sol, x_f_train):

        return torch.tensor(1.)

    def ubx0(self, t):
        type_BC = [DirichletBC()]
        if self.equation_type == "Rarefaction":
            out = torch.full(size=(t.shape[0], 1), fill_value=-1.0)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "Rarefaction_m":
            out = torch.full(size=(t.shape[0], 1), fill_value=0)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "sine" or self.equation_type == "gauss":
            out = torch.full(size=(t.shape[0], 1), fill_value=0)
            return out.reshape(-1, 1), type_BC
        else:
            out = torch.full(size=(t.shape[0], 1), fill_value=1.)
            return out.reshape(-1, 1), type_BC

    def ubx1(self, t):
        type_BC = [DirichletBC()]
        if self.equation_type == "Moving":
            out = torch.full(size=(t.shape[0], 1), fill_value=0.)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "Standing":
            out = torch.full(size=(t.shape[0], 1), fill_value=-1.)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "rar_s_standing":
            out = torch.full(size=(t.shape[0], 1), fill_value=1.)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "rar_s_moving":
            out = torch.full(size=(t.shape[0], 1), fill_value=1.)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "Rarefaction" or self.equation_type == "Rarefaction_m":
            out = torch.full(size=(t.shape[0], 1), fill_value=1.)
            return out.reshape(-1, 1), type_BC
        elif self.equation_type == "sine" or self.equation_type == "gauss":
            out = torch.full(size=(t.shape[0], 1), fill_value=0)
            return out.reshape(-1, 1), type_BC

    def uby0(self, t):
        type_BC = [DirichletBC()]
        if self.equation_type == "Standing":
            out = torch.full(size=(t.shape[0], 1), fill_value=-1.0)
            out_mod = torch.where(t[:, 1].unsqueeze(1) <= 0, torch.full_like(out, 1.0), out)
            return out_mod.reshape(-1, 1), type_BC
        elif self.equation_type == "Moving":
            out = torch.full(size=(t.shape[0], 1), fill_value=0.0)
            out_mod = torch.where(t[:, 1].unsqueeze(1) <= (t[:, 0].unsqueeze(1) / 2), torch.full_like(out, 1.0), out)
            return out_mod.reshape(-1, 1), type_BC
        elif self.equation_type == "Rarefaction":
            x1 = t[:, 1]
            t_values = t[:, 0]
            out_mod = torch.zeros_like(x1)
            for i in range(t.shape[0]):
                if t_values[i] == 0:
                    if x1[i] <= 0:
                        out_mod[i] = -1.0
                    else:
                        out_mod[i] = 1.0
                elif t_values[i] > 0:
                    if t_values[i] >= x1[i] >= -t_values[i]:
                        out_mod[i] = x1[i] / t_values[i]
                    elif x1[i] > t_values[i]:
                        out_mod[i] = 1.0
                    elif x1[i] < -t_values[i]:
                        out_mod[i] = -1.0
            return out_mod.reshape(-1, 1), type_BC
        elif self.equation_type == "sine" or self.equation_type == "gauss":
            out = torch.full(size=(t.shape[0], 1), fill_value=0)
            return out, type_BC

    def u0(self, x):
        if self.equation_type == "Moving":
            u = torch.full(size=(x.shape[0], 1), fill_value=0.)
            u_mod = torch.where(x[:, 0].unsqueeze(1) <= 0, torch.full_like(u, 1.0), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "Standing":
            u = torch.full(size=(x.shape[0], 1), fill_value=-1.0)
            u_mod = torch.where(x[:, 0].unsqueeze(1) <= 0, torch.full_like(u, 1.0), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "rar_s_standing":
            u = torch.ones_like(x)
            u_mod = torch.where((x > -0.5) & (x < 0.5), torch.tensor(-1.), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "rar_s_moving":
            u = torch.ones_like(x)
            u_mod = torch.where((x > -0.5) & (x < 0.5), torch.tensor(0.), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "Rarefaction":
            u = torch.full(size=(x.shape[0], 1), fill_value=1.0)
            u_mod = torch.where(x[:, 0].unsqueeze(1) <= 0, torch.full_like(u, -1.0), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "Rarefaction_m":
            u = torch.ones_like(x)
            u_mod = torch.where(x <= 0, torch.tensor(0.), u)
            return u_mod.reshape(-1, 1)
        elif self.equation_type == "sine":
            x1 = x[:, 0].unsqueeze(1)  
            x2 = x[:, 1].unsqueeze(1)  
            u = -torch.sin(pi * x1) * torch.cos(0.5 * pi * x2)
            return u.reshape(-1, 1)
        elif self.equation_type == "gauss":
            u = torch.exp(-x ** 2 / (0.2 ** 2))
            return u.reshape(-1, 1)


    def interpolate_f(self, file_ex, nu):
        exact_solution = np.loadtxt(file_ex, dtype=np.float64)
        condition = (exact_solution[:, 2] == 1.0) & (exact_solution[:, 3] == nu)

        time_column = exact_solution[condition, 0]  
        space_column = exact_solution[condition, 1] 
        exact_values = exact_solution[condition, -1]

        interpolator = NearestNDInterpolator((time_column, space_column), exact_values)

        def f(t_query, x_query):
            return interpolator(t_query, x_query)

        return f

    def exact(self, inputs):
        sol = torch.zeros_like(inputs[:, 0])

        if self.equation_type == "Moving":

            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1:]
                t_i = inputs[i, 0]
                x_i_1 = inputs[i, 1]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                else:
                    if x_i_1 <= 0.5 * t_i:
                        sol[i] = 1
                    else:
                        sol[i] = 0
        elif self.equation_type == "Standing":
            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1:]
                t_i = inputs[i, 0]
                x_i_1 = inputs[i, 1]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                else:
                    if x_i_1 <= 0.:
                        sol[i] = 1
                    else:
                        sol[i] = -1
        elif self.equation_type == "rar_s_moving":
            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1]
                t_i = inputs[i, 0]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                else:
                    if x_i <= 0.5 * t_i - 0.5:
                        sol[i] = 1
                    if x_i >= 0.5 * t_i - 0.5 and (x_i < 0.5):
                        sol[i] = 0
                    elif (x_i >= 0.5) and (x_i < 0.5 + t_i):
                        sol[i] = (x_i - 0.5) / t_i
                    elif x_i >= t_i + 0.5:
                        sol[i] = 1
        elif self.equation_type == "rar_s_standing":
            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1]
                t_i = inputs[i, 0]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                else:
                    if (x_i < -0.5) and (x_i >= -1):
                        sol[i] = 1

                    elif (x_i < 0.5 - t_i) and (x_i >= -0.5):
                        sol[i] = -1
                    elif (x_i >= 0.5 - t_i) and (x_i < 0.5 + t_i):
                        sol[i] = (x_i - 0.5) / t_i
                    elif x_i >= t_i + 0.5:
                        sol[i] = 1
        elif self.equation_type == "Rarefaction":
            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1:]
                t_i = inputs[i, 0]
                x_i_1 = inputs[i, 1]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                if (x_i_1 <= t_i) and (x_i_1 >= -t_i):
                    sol[i] = x_i_1 / t_i

                elif x_i_1 < -t_i:
                    sol[i] = -1
                elif x_i_1 > t_i:
                    sol[i] = 1
        elif self.equation_type == "Rarefaction_m":
            for i in range(inputs.shape[0]):
                x_i = inputs[i, 1]
                t_i = inputs[i, 0]
                if t_i == 0:
                    sol[i] = self.u0(x_i).reshape(1, )
                if (x_i <= t_i) and (x_i >= 0):
                    sol[i] = x_i / t_i
                elif x_i < 0:
                    sol[i] = 0
                elif x_i > t_i:
                    sol[i] = 1
        elif self.equation_type == "sine":

            f = self.interpolate_f("/root/ZHF/cuda0/Data/BurgersExact.txt", self.nu)

            for i in range(inputs.shape[0]):
                t_i = inputs[i, 0].item() 
                x_i = inputs[i, 1].item()  
                y_i = inputs[i, 2].item()  

               
                if t_i == 0:

                    sol[i] = self.u0(torch.tensor([x_i, y_i])).reshape(1, )

                else:

                    f_val = f(t_i, x_i)
                    cos_pi_y = np.cos(0.5 * pi * y_i)
                    sol[i] = f_val * cos_pi_y

        return sol.reshape(-1, 1)

    def compute_generalization_error(self, model, extrema=None, images_path=None):
        model = model.to(torch.device('cuda:0'))  
        model.eval()

        device = torch.device('cuda:0') 

        
        test_inp = self.convert(torch.rand((100000, 3)).to(device), self.domain_boundaries).to(device)

        Exact = self.exact(test_inp).detach().cpu().numpy().reshape(-1, 1)

        test_out = model(test_inp).detach().cpu().numpy()

        L2_test = np.mean(np.abs(Exact - test_out))
        print("Error Test:", L2_test)

        mean_exact = np.mean(np.abs(Exact))
        if mean_exact == 0:
            print("mean_exact is zero, setting relative L2 test to NaN")
            rel_L2_test = np.nan
        else:
            rel_L2_test = L2_test / mean_exact

        print("Relative Error Test:", rel_L2_test)
        return L2_test, rel_L2_test

    import torch
    import numpy as np
    import math

    def plotting(self, model, images_path, plot_test):
        model.cpu()
        model = model.eval()

        time_steps = [0.0, 0.25, 0.45]
        scale_vec = np.linspace(0.65, 1.55, len(time_steps))

        if not plot_test:
            plt.figure()
            plt.grid(True, which="both", ls=":")
            for val, scale in zip(time_steps, scale_vec):
                x = torch.linspace(-1, 1, 1000).reshape(-1, 1)
                t = torch.full(size=(x.shape[0], 1), fill_value=val)
                inputs = torch.cat([t, x], 1)
                ex = self.exact(inputs)
                x_plot = inputs[:, 1].reshape(-1, 1)
                plt.plot(x_plot.cpu().detach().numpy(), ex, linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$',
                         color=self.lighten_color('grey', scale), zorder=0)
                plt.scatter(x.cpu().detach().numpy(), model(inputs).cpu().detach().numpy(),
                            label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                            color=self.lighten_color('C0', scale),
                            zorder=10)

            plt.xlabel(r'$x$')
            plt.ylabel(r'u')
            plt.legend()
            plt.savefig(images_path + "/Samples.png", dpi=500)

        else:
            plt.figure()
            plt.grid(True, which="both", ls=":")
            for val, scale in zip(time_steps, scale_vec):
                x = torch.linspace(-1, 1, 100).reshape(-1, 1)
                t = torch.tensor(()).new_full(size=(x.shape[0], 1), fill_value=val)
                inputs_m = torch.cat([t, x], 1)
                w = self.fun_w(inputs_m, self.domain_boundaries, self.time_dimensions)
                plt.plot(x.detach().numpy(),
                         model(inputs_m).reshape(-1, ).detach().numpy() * w.reshape(-1, ).detach().numpy(),
                         label=r'$\phi$, $t=$' + str(val) + r'$s$',
                         color=self.lighten_color("C0", scale), zorder=10)
                plt.plot(x.detach().numpy(), model(inputs_m).reshape(-1, ).detach().numpy(),
                         label=r'$\theta$, $t=$' + str(val) + r'$s$', color=self.lighten_color("C0", scale),
                         zorder=10)

            plt.xlabel(r'$x$')
            plt.ylabel(r'u')
            plt.legend()
            plt.savefig(images_path + "/Samples_test.png", dpi=500)
