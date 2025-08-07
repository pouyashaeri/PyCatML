# Adjoint-based Optimizer
import torch
from torch.optim import Adam

class CategoricalOptimizer:
    def __init__(self, parameters, lr=1e-3):
        self.params = list(parameters)
        self.optimizer = Adam(self.params, lr=lr)

    def step(self, loss_fn):
        self.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_params(self, grads):
        """Apply parameter updates given gradient dict (optional low-level version)"""
        with torch.no_grad():
            for p, g in zip(self.params, grads):
                p -= g

    def as_adjoint_transformation(self):
        """Optional: Return as a categorical natural transformation between parameter states"""
        return lambda theta: [p.detach().clone() for p in self.params]
