import numpy as np
import math
import torch
import h5py
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_s(num_S, Nt, t_max=0.5):
    t = np.linspace(1e-4, t_max, Nt)
    random.seed(99)
    S_t = np.zeros((num_S, Nt))

    increase_factor = 100
    increase = np.linspace(2, 10, increase_factor)
    interval = np.ceil(num_S / increase_factor)
    j = 1
    for i in range(num_S):
        type = random.randint(1, 8)
        if type == 1:
            a = 2 * random.random()
            f = lambda t: (np.abs(np.sin(a * 2 * np.pi * t)) + 1)
        elif type == 2:
            a = 2 * random.random() - 1
            f = lambda t: np.exp(a * t) + 0.4
        elif type == 3:
            a = 2 * np.random.rand() - 1
            if a > 0:
                f = lambda t: a * t + 1
            else:
                f = lambda t: a * t + 1.5
        elif type == 4:
            a = 2 * np.random.rand() - 1
            f = lambda t: 0.5*(t - a) ** 2 + 1
        elif type == 5:
            a = 2 * np.random.rand() - 1
            f = lambda t: 0.3333 * np.abs((t - a) ** 3) + 1
        elif type == 6:
            a = 20 * np.random.rand()
            f = lambda t: 0.3*np.arctan(a * t) + 1
        elif type == 7:
            mu = np.random.rand()
            sigma = 0.2 * np.random.rand()
            f = lambda t: np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + 1
        elif type == 8:
            a = 2 * np.random.rand(2) - 1
            f = lambda t: abs(a[0]) / (a[1] * t + 1.5) + 1

        S_t[i, :] = np.array([f(ti) for ti in t])
        if np.floor((i+1) / interval) == j:
            S_t[int((j - 1) * interval):int(j * interval), :] *= increase[j-1]
            j += 1

    np.random.shuffle(S_t)
    S_t = torch.tensor(S_t).to(torch.float32)

    return S_t


def Transport_solve(s, x_position=0.5, Nmu=14, Nmu_half=7, Nx=51, t_max=0.5):
    s.to(device)
    Ns = s.shape[0]
    Nt = s.shape[1]
    dett = (t_max - 0.) / (Nt - 1)
    detx = (1 - 0.) / (Nx - 1)
    mu, w = np.polynomial.legendre.leggauss(Nmu)
    mu = torch.flip(torch.from_numpy(mu.astype(np.float32)), dims=[0]).to(device)
    w = torch.flip(torch.from_numpy(w.astype(np.float32)), dims=[0]).to(device)

    v = 29.98
    a = 0.01372
    sigma = 10  # sigma/T^3
    Cv = 1
    epsilon = 1

    I_tbound = 0.5 * a * v * 1 ** 4 * torch.ones(Ns, Nmu, Nx, device=device)
    T_tbound = 1 * torch.ones(Ns, Nx - 1, device=device)
    I = torch.zeros(Ns, Nmu, Nx, Nt, device=device)
    I[:, :, :, 0] = I_tbound
    T = torch.ones(Ns, Nt, Nx - 1, device=device)
    T[:, -1, :] = T_tbound

    # mu<0
    I_xbound_nega = s.unsqueeze(1).expand(-1, Nmu_half, -1).clone().to(device)
    I_xbound_nega[:, :, 0] = I_tbound[:, Nmu_half:, -1]

    I_last = I_tbound
    pn = torch.einsum('bij,i->bj', I_tbound[:, :, 1:], w)
    pm = pn
    for ti in range(Nt - 1):
        I_tnode = torch.zeros(Ns, Nmu, Nx, device=device)
        I_tnode[:, Nmu_half:, -1] = I_xbound_nega[:, :, ti+1]
        T_num = 1
        T_error = 1
        Tn = 0.95 * T[:, Nt - ti - 2, :]
        while T_error >= 1e-5:
            I_num = 1
            I_error = 1
            while I_error >= 1e-6:
                Ttx = torch.ones(Ns, Nx - 1, device=device)
                Stx = torch.zeros(Ns, Nx - 1, device=device)
                for xi in range(Nx - 1):
                    Ttx[:, Nx-2-xi] = (sigma/Tn[:, Nx-2-xi]**3 * pn[:, Nx-2-xi] + 3 * sigma/Tn[:, Nx-2-xi]**3 * a * v * Tn[:, Nx-2-xi]**4
                                      + epsilon**2 * Cv / dett * T[:, Nt-ti-1, Nx-2-xi]) / \
                                          (epsilon**2 * Cv / dett + 4 * sigma/Tn[:, Nx-2-xi]**3 * a * v * Tn[:, Nx-2-xi]**3)
                    Stx[:, Nx-2-xi] = abs(detx * 0.5 * sigma/Tn[:, Nx-2-xi]**3 * a * v *
                                      (4 * Tn[:, Nx-2-xi]**3 * Ttx[:, Nx-2-xi] - 3 * Tn[:, Nx-2-xi]**4))
                    # mu<0
                    for mi in range(Nmu_half, Nmu):
                        I_tnode[:, mi, Nx-2-xi] = (epsilon**2 * detx / (v*dett) * I_last[:, mi, Nx-2-xi] + Stx[:, Nx-2-xi]
                                                  - epsilon * mu[mi] * I_tnode[:, mi, Nx-1-xi]) / \
                                                        (epsilon**2 * detx / (v*dett) - epsilon*mu[mi] + sigma/Tn[:, Nx-2-xi]**3 *detx)

                # mu>0
                I_tnode[:, :Nmu_half, 0] = torch.flip(I_tnode[:, Nmu_half:, 0], dims=[1])
                for xi in range(Nx-1):
                    for mi in range(Nmu_half):
                        I_tnode[:, mi, xi+1] = (epsilon**2 * detx / (v*dett) * I_last[:, mi, xi+1] + Stx[:, xi]
                                               + epsilon * mu[mi] * I_tnode[:, mi, xi]) / \
                                               (epsilon**2 * detx / (v*dett) + epsilon*mu[mi] + sigma/Tn[:, xi]**3 * detx)

                # update Ttx & pn
                for xi in range(Nx-1):
                    pm[:, xi] = I_tnode[:, :Nmu_half, xi+1] @ w[:Nmu_half] + I_tnode[:, Nmu_half:, xi] @ w[Nmu_half:]
                I_error = torch.max(torch.abs(pm - pn))
                pn = pm
                I_num = I_num + 1

            T_error = torch.max(torch.abs(Ttx - Tn))
            Tn = Ttx
            T_num = T_num + 1

        print(ti+1)
        I_last = I_tnode
        T[:, Nt-ti-2] = Tn
        I[:, :, :, ti+1] = I_tnode

    I_position = torch.squeeze(I[:, :, int(x_position/detx)-1, :])
    F_t = torch.einsum('bij,i->bj', I_position, mu*w)

    return I_position, F_t


def generate_and_save_data():
    S_t = make_s(num_S=100000, Nt=64, t_max=0.5).to(device)

    print('S_t exist nan: ', torch.isnan(S_t).any())
    print('S_t exist inf: ', torch.isinf(S_t).any())
    print('shape of S_t', S_t.size())
    print('min of S_t', S_t.min())
    print('max of S_t', S_t.max())
    print('mean of S_t', S_t.mean())
    print('std of S_t', S_t.std())

    I_position, F_t = Transport_solve(S_t, x_position=0.5, Nmu=14, Nmu_half=7, Nx=51, t_max=0.5)
    print('F_t exist nan: ', torch.isnan(F_t).any())
    print('F_t exist inf: ', torch.isinf(F_t).any())
    print('shape of F_t', F_t.size())
    print('min of F_t', F_t.min())
    print('max of F_t', F_t.max())
    print('mean of F_t', F_t.mean())
    print('std of F_t', F_t.std())

    num_train = math.ceil(S_t.shape[0] * 0.9)
    data = torch.cat((S_t.unsqueeze(1), I_position, F_t.unsqueeze(1)), dim=1)
    data_cpu = data.cpu()

    with h5py.File('./data/Transport_Train.h5', 'w') as f1:
        group_train = f1.create_group('Train')
        group_train.create_dataset('train', data=data_cpu[:num_train, ...])

    with h5py.File('./data/Transport_Test.h5', 'w') as f1:
        group_train = f1.create_group('Test')
        group_train.create_dataset('test', data=data_cpu[num_train:, ...])


if __name__ == '__main__':
    generate_and_save_data()