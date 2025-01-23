import math
import torch
import torch.nn.functional as F
import pandas as pd
from collections import namedtuple
from torch import nn
from torch.cuda.amp import autocast
from einops import reduce
from tqdm.auto import tqdm

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            beta_schedule='cosine',
            rescaler=1,
            is_condition_bc=True,
            is_condition_ft=True,
            train_on_padded_locations=False,
    ):

        super().__init__()

        self.model = model
        self.channels = self.model.channels
        self.rescaler = rescaler

        assert type(seq_length) is tuple and len(seq_length) == 2
        self.traj_size = seq_length

        self.num_timesteps = int(timesteps)
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps).to(device)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_prev = F.pad(alphas[:-1], (1, 0), value=1.)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        self.alphas = alphas.to(torch.float32).clone()
        self.alphas_prev = alphas_prev.to(torch.float32).clone()
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1. - alphas_cumprod)
        # loss_weight = betas / (alphas * (1. - alphas_cumprod_prev) + 1e-8)
        loss_weight = torch.ones_like(snr)
        register_buffer('loss_weight', loss_weight)

        self.is_condition_bc = is_condition_bc
        self.is_condition_ft = is_condition_ft
        self.train_on_padded_locations = train_on_padded_locations

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    def model_predictions(self, x, t):
        with torch.no_grad():
            pred_noise = self.model(x, t)
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start

        x_start.clamp_(-1., 1.)

        model_mean, posterior_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, x_start, preds.pred_noise

    def p_sample(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        noise = torch.randn_like(x) if t > 0 else 0.
        # model_mean, posterior_variance, x_start, pred_noise = self.p_mean_variance(x, batched_times)
        # pred_img = model_mean + torch.sqrt(posterior_variance) * noise

        C1 = torch.sqrt(1. / self.alphas)
        C2 = (1 - self.alphas) / (self.sqrt_one_minus_alphas_cumprod * torch.sqrt(self.alphas))
        posterior_variance = extract(self.posterior_variance, batched_times, x.shape)
        pred_noise = self.model(x, batched_times)
        pred_img = (extract(C1, batched_times, x.shape) * x - extract(C2, batched_times, x.shape) * pred_noise) \
                    + torch.sqrt(posterior_variance) * noise
        # print(pred_img.max())
        # print(pred_img.min())
        return pred_img, _, pred_noise

    # removed no_grad decorator here
    def p_sample_loop(self, dataset, shape, rescaler):
        batch, device = shape[0], self.betas.device

        img_valid = torch.randn(shape, device=device)
        img_test = torch.randn((3, 1, 16, 64), device=device)
        img = torch.cat((img_valid, img_test), dim=0)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.is_condition_bc:
                img[:batch, :, 1:-1, 0] = 2 * (dataset[:, :, 1:-1, 0] / rescaler) - 1
                img[batch:, :, 1:-1, 0] = 2 * (0.5 * 0.01372 * 29.98 * 1 ** 4 / rescaler) - 1

            if self.is_condition_ft:
                img[:batch, :, -1, :] = 2 * (dataset[:, :, -1, :] / rescaler) - 1
                tt = torch.linspace(0, 0.5, 64).reshape((1, 64))
                gt = pd.read_csv('./data/Ft.csv', header=None).to_numpy()
                gt = torch.from_numpy(gt)
                img[batch:, 0, -1, :] = 2 * (gt / rescaler) - 1

            img_curr, _, pred_noise = self.p_sample(img, t)
            img = img_curr.detach()

        return img

    def sample(self, dataset, batch_size, rescaler):
        sample_size = (batch_size, self.channels, *self.traj_size)
        return self.p_sample_loop(dataset, sample_size, rescaler)

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.is_condition_bc:
            x_t[:, :, 1:-1, 0] = x_start[:, :, 1:-1, 0]

        if self.is_condition_ft:
            x_t[:, :, -1, :] = x_start[:, :, -1, :]

        model_out = self.model(x_t, t)

        target = noise

        if self.is_condition_bc:
            target[:, :, 1:-1, 0] = 0

        if self.is_condition_ft:
            target[:, :, -1, :] = 0

        # if not self.train_on_padded_locations:
        #     model_out[:, :, N:, :] = target[:, :, N:, :]

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_total = loss.mean()

        return loss_total

    def forward(self, img):
        b, c, nmu, nx = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        device = img.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t)
