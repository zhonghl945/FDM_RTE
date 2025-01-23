import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from Transport_ZHL import Transport_solve
from Trainer import Trainer
from Main_Train import get_2d_ddpm, rescaler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cosine_beta_J_schedule(t, s=0.008):
    timesteps = 1000
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)[t]


def load_2dconv_model():
    dataset = get_dataset(mode='Test')
    ddpm = get_2d_ddpm(rescaler)

    trainer = Trainer(
        ddpm,
        dataset,
        device=device,
        train_batch_size=16,
        results_folder='./trained_models/',
        train_num_steps=100000,
        save_and_sample_every=10000,
        record_step=100,
        train_lr=1e-4,
    )
    trainer.load(10)

    return ddpm


def get_dataset(mode='Train'):
    if mode == 'Train':
        with h5py.File('./data/Transport_Train.h5', 'r') as file:
            group_train = file['Train']
            data = group_train['train'][:]

        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(1).expand(-1, 1, -1, -1).clone()

        rescaler_train = data.abs().max()
        data = 2 * (data / rescaler_train) - 1
        return data, rescaler_train

    elif mode == 'Test':
        with h5py.File('./data/Transport_Test.h5', 'r') as file:
            group_test = file['Test']
            data = group_test['test'][:]

        data = torch.tensor(data, dtype=torch.float32)
        data = data.unsqueeze(1).expand(-1, 1, -1, -1).clone()
        return data

    else:
        raise ValueError('Bad data mode')


def diffuse_2dconv(dataset, batch_size, rescaler):
    ft_from_x = lambda x: x[:, 0, -1, :]
    st_from_x = lambda x: x[:, 0, 0, :]

    ddpm = load_2dconv_model()

    t_max = 0.5
    img = (ddpm.sample(dataset, batch_size, rescaler) + 1) * rescaler / 2
    _, ft_s = Transport_solve(st_from_x(img), x_position=0.5, t_max=t_max)

    "valid set"
    # L2 error between generated f(t) and solved f(t) by generated s(t)
    ddpm_mse = (ft_from_x(img[:batch_size]) - ft_s[:batch_size]).square().mean((-1, -2))

    # L2 error between true f(t) and solved f(t) by generated s(t)
    J_mse = (torch.squeeze(dataset[:, 0, -1, :]) - ft_s[:batch_size]).square().mean((-1))
    J_mae = (torch.squeeze(dataset[:, 0, -1, :]) - ft_s[:batch_size]).abs().mean((-1))

    print('Valid set: ddpm_mse', ddpm_mse.mean(0))
    print('Valid set: J_mse', J_mse.mean(0))
    print('Valid set: J_mae', J_mae.mean(0))

    "test set"
    tt = torch.linspace(0, 0.5, 64).reshape((1, 64))
    S1 = (3 * torch.ones(1, 64))
    S2 = (10 ** tt + 1)
    S3 = -torch.log(0.1 * (tt + 0.1))
    gt = pd.read_csv('./data/Ft.csv', header=None).to_numpy()
    gt = torch.from_numpy(gt)

    # L2 error between true f(t) and solved f(t) by generated s(t)
    J_mse_test = (gt - ft_s[batch_size:].cpu()).square().mean((-1))
    J_relative_mse_test = J_mse_test / gt.cpu().square().mean((-1))

    print('Test Set: J_L2_error ', torch.sqrt(J_mse_test).mean(0))
    print('Test Set: J_relative_L2_error ', torch.sqrt(J_relative_mse_test).mean(0))

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, t_max, 64), ft_s[-3, :].cpu().numpy(), label='$F_1(t)$_ $Solve$', linestyle='--',
             color='red', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), ft_s[-2, :].cpu().numpy(), label='$F_2(t)$_ $Solve$', linestyle='--',
             color='black', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), ft_s[-1, :].cpu().numpy(), label='$F_3(t)$_ $Solve$', linestyle='--',
             color='orange', linewidth=2.5)

    plt.plot(np.linspace(0, t_max, 64), gt[0, :].numpy(), label='$F_1(t)$_ $Target$', color='red', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), gt[1, :].numpy(), label='$F_2(t)$_ $Target$', color='black', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), gt[2, :].numpy(), label='$F_3(t)$_ $Target$', color='orange', linewidth=2.5)
    plt.legend()
    plt.xlabel(r'$t$', fontsize=20)
    plt.ylabel(r'$F(t)$', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('Control.png', format='png', dpi=300)

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, t_max, 64), st_from_x(img)[-3, :].cpu().numpy(), label='$S_1(t)$_ $Pre$', linestyle='--',
             color='red', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), st_from_x(img)[-2, :].cpu().numpy(), label='$S_2(t)$_ $Pre$', linestyle='--',
             color='black', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), st_from_x(img)[-1, :].cpu().numpy(), label='$S_3(t)$_ $Pre$', linestyle='--',
             color='orange', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), (S1.squeeze()).numpy(), label='$S_1(t)$_ $Best$', color='red', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), (S2.squeeze()).numpy(), label='$S_2(t)$_ $Best$', color='black', linewidth=2.5)
    plt.plot(np.linspace(0, t_max, 64), (S3.squeeze()).numpy(), label='$S_3(t)$_ $Best$', color='orange', linewidth=2.5)

    plt.legend()
    plt.xlabel(r'$t$', fontsize=20)
    plt.ylabel(r'$S(t)$', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('Prediction.png', format='png', dpi=300)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if not os.path.exists('./data/ft.csv'):
        # generalized test set
        tt = torch.linspace(0, 0.5, 64).reshape((1, 64))
        S1 = 3 * torch.ones(1, 64)
        S2 = 10 ** tt + 1
        S3 = -torch.log(0.1 * (tt + 0.1))
        S = torch.cat((S1, S2, S3), dim=0)
        _, Ft = Transport_solve(S, x_position=0.5, Nmu=14, Nmu_half=7, Nx=51, t_max=0.5)
        # plt.plot(np.linspace(0, 0.5, 64), Ft[0, :].cpu().numpy(), label='F1(t)_solve', linestyle='--', color='red')
        # plt.plot(np.linspace(0, 0.5, 64), Ft[1, :].cpu().numpy(), label='F2(t)_solve', linestyle='--', color='green')
        # plt.plot(np.linspace(0, 0.5, 64), Ft[2, :].cpu().numpy(), label='F2(t)_solve', linestyle='--', color='blue')
        # plt.show()
        df = pd.DataFrame(Ft.cpu().numpy()).to_csv('./data/Ft.csv', index=False, header=False)

    test_dataset = get_dataset(mode='Test')
    batch_size = 3
    indices = torch.randperm(test_dataset.size(0))[:batch_size]
    dataset = test_dataset[indices, ...].to(device)

    diffuse_2dconv(dataset=dataset, batch_size=batch_size, rescaler=rescaler)

