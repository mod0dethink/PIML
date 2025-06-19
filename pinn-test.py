import torch
import torch.nn as nn

# 1D PINN model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        input = torch.cat([x, t], dim=1)
        return self.net(input)

# PDE: ∂u/∂t = α ∂²u/∂x²
def pde_residual(model, x, t, alpha):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    
    return u_t - alpha * u_xx
