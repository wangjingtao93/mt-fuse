"""
利用torch.autograd计算标量函数y=x^3 + sin(x)在x=1,pi,和5时的一阶段导和二阶导
"""

import torch
import numpy as np

x = torch.tensor([1, np.pi, 5],requires_grad=True)
y = x**3 + torch.sin(x)

dy = 3*x**2 + torch.cos(x)

d2y = 6*x - torch.sin(x)

dydx = torch.autograd.grad(y,x, grad_outputs=torch.ones(x.shape), # #注意这里需要人为指定
                           create_graph=True, 
                             retain_graph=True)# 为了计算二阶导

