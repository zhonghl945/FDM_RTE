import torch
import numpy as np
import h5py

from Unet2D import Unet2D
from GaussianDiffusion import GaussianDiffusion
from Trainer import Trainer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_2d_ddpm(rescaler):
    u_net = Unet2D(
        dim=64,
        out_dim=1,
        dim_mults=[1, 2, 4],
        attn_heads=4,
        attn_dim_head=64,
        channels=1,
        resnet_block_groups=1,
    )

    ddpm = GaussianDiffusion(
        u_net,
        seq_length=(16, 64),
        timesteps=1000,
        beta_schedule='cosine',
        rescaler=rescaler,
        is_condition_bc=True,
        is_condition_ft=True,
    )

    return ddpm


def run_2d_Unet(dataset, rescaler):
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
    # trainer.load(10)  # load pervious file
    trainer.train()


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


_, rescaler = get_dataset(mode='Train')
print(f'\nRescaling data by rescaler: {rescaler}')

if __name__ == "__main__":
    dataset, _ = get_dataset(mode='Train')
    print(f'\ndata shape: {dataset.shape}')

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    run_2d_Unet(dataset, rescaler)
