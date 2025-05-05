class DirichletBC:
    def __init__(self):
        pass
        
    def apply(self, model, x_boundary, u_boundary, n_out):
        predictions = model(x_boundary)
        predicted_values = predictions[:, n_out]
        target_values = u_boundary[:, n_out]
        
        return predicted_values, target_values
