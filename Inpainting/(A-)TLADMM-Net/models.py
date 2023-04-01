import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigvalsh

##################################################

# # #            Adaptive-LISTA              # # #

##################################################


class Adaptive_ISTA(nn.Module):
    def __init__(self, n, m, D=None, T=6, lambd=1.0):
        super(Adaptive_ISTA, self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.W1 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W2 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W1.weight.data = torch.eye(n)
        self.W2.weight.data = torch.eye(n)
        # ISTA Stepsizes
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        # Initialization
        if D is not None:
            L = 5  # float(eigvalsh(D.T @ D, eigvals=(m - 1, m - 1)))
            self.etas.data *= 1 / L
            self.gammas.data *= 1 / L
        self.reinit_num = 0  # Number of re-initializations

    def _A(self, D, i):
        A_tmp = self.W1.weight @ D
        return self.gammas[i, :, :, :] * A_tmp.transpose(1, 2)

    def _B(self, D, i):
        B_tmp = self.W2.weight @ D
        return self.gammas[i, :, :, :] * B_tmp.transpose(1, 2) @ B_tmp

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, D):
        y = y.unsqueeze(2)
        x = torch.zeros(y.shape[0], self.m, y.shape[2])
        if y.is_cuda:
            x = x.cuda()
        for i in range(0, self.T + 1):
            x = self._shrink(x - self._B(D, i) @ x + self._A(D, i) @ y, self.etas[i, :, :, :])
        return x.squeeze()

    def reinit(self):
        reinit_num = self.reinit_num + 1
        self.__init__(n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd)
        self.reinit_num = reinit_num


##################################################

# # #           Adaptive-LFISTA              # # #

##################################################


class Adaptive_FISTA(Adaptive_ISTA):
    def forward(self, y, D):
        y = y.unsqueeze(2)
        t_curr = 1
        z = torch.zeros(y.shape[0], self.m, y.shape[2])
        x_curr = torch.zeros(y.shape[0], self.m, y.shape[2])
        if y.is_cuda:
            z, x_curr = z.cuda(), x_curr.cuda()
        for i in range(0, self.T + 1):
            x_prev = x_curr
            x_curr = self._shrink(z - self._B(D, i) @ z + self._A(D, i) @ y, self.etas[i, :, :, :])

            t_prev = t_curr
            t_curr = 0.5 * (1 + np.sqrt(1 + 4 * (t_prev ** 2)))
            z = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)
        return x_curr.squeeze()


##################################################

# # #       Adaptive-LISTA Reverse Order     # # #

##################################################


class Adaptive_ISTA_Rev(nn.Module):
    def __init__(self, n, m, D=None, T=6, lambd=1.0):
        super(Adaptive_ISTA_Rev, self).__init__()
        self.n, self.m = n, m
        self.T = T
        self.lambd = lambd 
        self.D = D.cuda()  
        self.W1 = nn.Linear(m, n, bias=False)  
        self.etas = nn.Parameter(0.1*torch.ones(T + 1))
        self.gammas = nn.Parameter(0.1*torch.ones(T + 1))
        self.L_step = nn.Parameter(torch.ones(T + 1))
        self.rho = nn.Parameter(0.1*torch.ones(T + 1)) 
        self.u = nn.Parameter(torch.ones(T + 1)) 
        self.h = nn.Parameter(torch.ones(T + 1)) 
        self.theta = nn.Parameter(0.4*torch.ones(T + 1))
        self.thetax = nn.Parameter(0.1*torch.ones(T + 1))
        self.thetaL = nn.Parameter(0.1*torch.ones(T + 1))
        # Initialization
        if D is not None:
            self.W1.weight.data = D
        self.reinit_num = 0  

    def _A(self, i, Phi):
        A_tmp = Phi @ self.W1.weight
        return self.gammas[i] * A_tmp.transpose(1, 2)

    def _B(self, i, Phi):
        B_tmp = Phi @ self.W1.weight
        return self.gammas[i] * B_tmp.transpose(1, 2) @ (Phi @ self.D)

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, Phi):
        y = y.unsqueeze(2)
        x = self._shrink(self._A(0, Phi) @ y, self.etas[0, :, :, :])
        for i in range(1, self.T + 1):
            x = self._shrink(x - self._B(i, Phi) @ x + self._A(i, Phi) @ y, self.etas[i, :, :, :])
        return x.squeeze()

    def reinit(self):
        reinit_num = self.reinit_num + 1
        self.__init__(n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd)
        self.reinit_num = reinit_num

class Adaptive_ISTA_Rev_original(nn.Module):
    def __init__(self, n, m, D=None, T=6, lambd=1.0):
        super(Adaptive_ISTA_Rev, self).__init__()
        self.n, self.m = n, m
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.D = D  # D with which the sparse representations were created
        self.W1 = nn.Linear(m, n, bias=False)  # Weight Matrix
        self.W2 = nn.Linear(m, n, bias=False)  # Weight Matrix
        # ISTA Stepsizes, eta = 1/L
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        # Initialization
        if D is not None:
            L = float(eigvalsh(D.T @ D, eigvals=(m - 1, m - 1)))
            self.etas.data *= 1 / L
            self.gammas.data *= 1 / L
            self.W1.weight.data = D
            self.W2.weight.data = D
        else:
            self.W1.weight.data = 1 / (self.T + 1) * self.W1.weight.data
            self.W2.weight.data = 1 / np.sqrt(self.T + 1) * self.W2.weight.data
        self.reinit_num = 0  # Number of re-initializations

    def _A(self, i, Phi):
        A_tmp = Phi @ self.W1.weight
        return self.gammas[i, :, :, :] * A_tmp.transpose(1, 2)

    def _B(self, i, Phi):
        B_tmp = Phi @ self.W2.weight
        return self.gammas[i, :, :, :] * B_tmp.transpose(1, 2) @ B_tmp

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, Phi):
        y = y.unsqueeze(2)
        x = self._shrink(self._A(0, Phi) @ y, self.etas[0, :, :, :])
        for i in range(1, self.T + 1):
            x = self._shrink(x - self._B(i, Phi) @ x + self._A(i, Phi) @ y, self.etas[i, :, :, :])
        return x.squeeze()

    def reinit(self):
        reinit_num = self.reinit_num + 1
        self.__init__(n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd)
        self.reinit_num = reinit_num

##################################################

# # #       Adaptive-FISTA Reverse Order     # # #

##################################################


class Adaptive_FISTA_Rev(Adaptive_ISTA_Rev_original):
    def forward(self, y, Phi):
        y = y.unsqueeze(2)  # [512,64,1]
        t_curr = 1
        z = torch.zeros(y.shape[0], self.m, y.shape[2])
        x_curr = torch.zeros(y.shape[0], self.m, y.shape[2])
        if y.is_cuda:
            z, x_curr = z.cuda(), x_curr.cuda()
        for i in range(0, self.T + 1):
            x_prev = x_curr
            x_curr = self._shrink(
                z - self._B(i, Phi) @ z + self._A(i, Phi) @ y, self.etas[i, :, :, :]
            )
            t_prev = t_curr
            t_curr = 0.5 * (1 + np.sqrt(1 + 4 * (t_prev ** 2)))
            z = x_curr + (t_prev - 1) / t_curr * (x_curr - x_prev)
        return x_curr.squeeze()
##################################################

# # #       D-LADMM      # # #

##################################################
class DLADMM(Adaptive_ISTA_Rev):
    def forward(self, y, Phi): 
        y = y.unsqueeze(2) 
        x = torch.zeros(y.shape[0], self.m, y.shape[2]) 
        z = torch.zeros(y.shape[0], self.n, y.shape[2])
        L = torch.zeros(y.shape[0], self.n, y.shape[2])
        if y.is_cuda:
            z, x, L = z.cuda(), x.cuda(), L.cuda()
        for i in range(0, self.T + 1):
            x = self._shrink(
                x - self.rho[i] * (self._B(i, Phi) @ x - self._A(i, Phi) @ y + self._A(i, Phi) @(z + L)), self.etas[i])
            z = - self.rho[i] * (Phi @ self.D @ x - y + L) / (2 + self.rho[i])
            L = L + self.L_step[i] * (z + Phi @ self.D @ x - y)
        return x.squeeze()
##################################################

# # #       ELADMM      # # #

##################################################
class EADMM(Adaptive_ISTA_Rev):
    def forward(self, y, Phi): 
        y = y.unsqueeze(2) 
        x = torch.zeros(y.shape[0], self.m, y.shape[2]) 
        z = torch.zeros(y.shape[0], self.n, y.shape[2])
        L = torch.zeros(y.shape[0], self.n, y.shape[2])
        if y.is_cuda:
            z, x, L = z.cuda(), x.cuda(), L.cuda()
        for i in range(0, self.T + 1):
            x = self._shrink(
                x - self.h[i] * self.rho[i] * (self._B(i, Phi) @ x - self._A(i, Phi) @ y + self._A(i, Phi) @(z + L)), self.etas[i])
            z = - self.rho[i] * (Phi @ self.D @ x - y + L) / (2 + self.rho[i])
            L = L + self.h[i] * self.L_step[i] * (z + Phi @ self.D @ x - y)
        return x.squeeze()
##################################################

# # #       TLADMM      # # #

##################################################    
class TADMM(Adaptive_ISTA_Rev):
    def forward(self, y, Phi): 
        y = y.unsqueeze(2) 
        x = torch.zeros(y.shape[0], self.m, y.shape[2]) 
        z = torch.zeros(y.shape[0], self.n, y.shape[2])
        L = torch.zeros(y.shape[0], self.n, y.shape[2])
        if y.is_cuda:
            z, x, L = z.cuda(), x.cuda(), L.cuda()
        for i in range(0, self.T + 1):
            x_pre = self._shrink(
                x - self.rho[i] * (self._B(i, Phi) @ x - self._A(i, Phi) @ y + self._A(i, Phi) @(z + L)), self.etas[i])
            x = self._shrink(x - self.u[i] * (self.rho[i] * (self._B(i, Phi) @ x - self._A(i, Phi) @ y + self._A(i, Phi) @ (z + L)) + 
                self.rho[i] * (self._B(i, Phi) @ x_pre - self._A(i, Phi) @ y + self._A(i, Phi) @ (z + L))), self.etas[i])
            z = - self.rho[i] * (Phi @ self.D @ x - y + L) / (2 + self.rho[i])
            L = L + self.L_step[i] * (z + Phi @ self.D @ x - y)
        return x.squeeze()
##################################################

# # #       A-ELADMM      # # #

##################################################  
class A_EADMM(Adaptive_ISTA_Rev):
    def forward(self, y, Phi): 
        y = y.unsqueeze(2) 
        x0 = torch.zeros(y.shape[0], self.m, y.shape[2]) 
        z0 = torch.zeros(y.shape[0], self.n, y.shape[2])
        L0 = torch.zeros(y.shape[0], self.n, y.shape[2])
        X = list()
        Z = list()
        L = list()
        if y.is_cuda:
            z0, x0, L0 = z0.cuda(), x0.cuda(), L0.cuda()
        for i in range(0, self.T + 1):
            if i == 0:
                X.append(self._shrink(
                    x0 - self.rho[i] * self._B(i, Phi) @ x0 + self.rho[i] * self._A(i, Phi) @ y - self.rho[i] * self._A(i, Phi) @ (z0 + L0), self.etas[i] #z是论文中的x
                ))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L0) / (2 + self.rho[i]))
                L.append(L0 + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
            elif i == 1:
                hatx = X[-1] + self.thetax[i] * (X[-1] - x0) 
                X.append(self._shrink(
                    hatx - self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @(Z[-1] + L[-1])), self.etas[i]))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L[-1]) / (2 + self.rho[i]))
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L0) 
                L.append(hatL + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
            else:
                hatx = X[-1] + self.thetax[i] * (X[-1] - X[-2]) 
                X.append(self._shrink(
                    hatx - self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @(Z[-1] + L[-1])), self.etas[i]))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L[-1]) / (2 + self.rho[i]))
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L[-2]) 
                L.append(hatL + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
        return X[-1].squeeze()

##################################################

# # #       A-TLADMM      # # #

##################################################    

class A_TADMM(Adaptive_ISTA_Rev):
    def forward(self, y, Phi):
        y = y.unsqueeze(2)
        x0 = torch.zeros(y.shape[0], self.m, y.shape[2])
        z0 = torch.zeros(y.shape[0], self.n, y.shape[2])
        L0 = torch.zeros(y.shape[0], self.n, y.shape[2])
        X = list()
        Z = list()
        L = list()
        if y.is_cuda:
            z0, x0, L0 = z0.cuda(), x0.cuda(), L0.cuda()
        for i in range(0, self.T + 1):
            if i == 0:
                X.append(self._shrink(
                    x0 - self.rho[i] * self._B(i, Phi) @ x0 + self.rho[i] * self._A(i, Phi) @ y - self.rho[i] * self._A(i, Phi) @ (z0 + L0), self.etas[i] #z是论文中的x
                ))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L0) / (2 + self.rho[i]))
                L.append(L0 + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
            elif i == 1:
                hatx = X[-1] + self.thetax[i] * (X[-1] - x0) 
                x_pre = self._shrink(
                    hatx - self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @(Z[-1] + L[-1])), self.etas[i])
                X.append(self._shrink(hatx - 0.5 * self.u[i] / self.theta[i] * (self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @ (Z[-1] + L[-1])) + 
                    self.rho[i] * (self._B(i, Phi) @ x_pre - self._A(i, Phi) @ y + self._A(i, Phi) @ (Z[-1] + L[-1]))), self.etas[i]))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L[-1]) / (2 + self.rho[i]))
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L0) 
                L.append(hatL + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
            else:
                hatx = X[-1] + self.thetax[i] * (X[-1] - X[-2]) 
                x_pre = self._shrink(
                    hatx - self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @(Z[-1] + L[-1])), self.etas[i])
                X.append(self._shrink(hatx - 0.5 * self.u[i] / self.theta[i] * (self.rho[i] * (self._B(i, Phi) @ hatx - self._A(i, Phi) @ y + self._A(i, Phi) @ (Z[-1] + L[-1])) + 
                    self.rho[i] * (self._B(i, Phi) @ x_pre - self._A(i, Phi) @ y + self._A(i, Phi) @ (Z[-1] + L[-1]))), self.etas[i]))
                Z.append(- self.rho[i] * (Phi @ self.D @ X[-1] - y + L[-1]) / (2 + self.rho[i]))
                hatL = L[-1] + self.thetaL[i] * (L[-1] - L[-2]) 
                L.append(hatL + self.L_step[i] * (Z[-1] + Phi @ self.D @ X[-1] - y))
        return X[-1].squeeze()

