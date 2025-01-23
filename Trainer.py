import torch
import datetime

from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            device=device,
            train_batch_size=16,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=10000,
            results_folder='./results',
            max_grad_norm=1.,
            record_step=1,
    ):
        super().__init__()

        self.device = device
        self.accelerator = Accelerator(split_batches=True, mixed_precision='no')
        self.model = diffusion_model

        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps
        self.record_step = record_step

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=0)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        make_dir(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'loss': self.total_loss,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.device

        if type(milestone) is int:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            data = torch.load(str(self.results_folder / milestone), map_location=device)

        print('loss: ', data['loss'])
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.step_initial = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        print(self.device)
        writer = SummaryWriter(logdir='tensorboard_runs/{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M-%S")))

        accelerator = self.accelerator
        device = self.device
        loss_record = []
        if self.step == 0:
            self.step_initial = 0

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                data = next(self.dl).to(device)

                with self.accelerator.autocast():
                    loss = self.model(data)
                    total_loss += loss.item()

                if (self.step+1) % self.record_step == 0:
                    loss_record.append(total_loss)

                self.accelerator.backward(loss)

                self.total_loss = total_loss
                pbar.set_description(f'loss: {total_loss:.7f}')
                writer.add_scalar('loss', total_loss, self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(f"Gradient of {name}: {param.grad.max()}")

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                        self.save(milestone)

                pbar.update(1)

        record_len = int((self.train_num_steps - self.step_initial) / self.record_step)
        interval = range(1, record_len+1)
        plt.figure(figsize=(10, 5))
        plt.yscale('log')
        plt.plot(interval, loss_record)

        choose_point10 = int((self.train_num_steps - self.step_initial) * 1 / self.record_step)
        plt.scatter(choose_point10, loss_record[choose_point10-1], color='red')
        choose_point9 = int((self.train_num_steps - self.step_initial) * 0.9 / self.record_step)
        plt.scatter(choose_point9, loss_record[choose_point9-1], color='red')
        choose_point8 = int((self.train_num_steps - self.step_initial) * 0.8 / self.record_step)
        plt.scatter(choose_point8, loss_record[choose_point8-1], color='red')
        choose_point8 = int((self.train_num_steps - self.step_initial) * 0.7 / self.record_step)
        plt.scatter(choose_point8, loss_record[choose_point8-1], color='red')
        choose_point8 = int((self.train_num_steps - self.step_initial) * 0.6 / self.record_step)
        plt.scatter(choose_point8, loss_record[choose_point8-1], color='red')

        plt.title(' Log Scale Loss Over Interval')
        plt.xlabel(f'interval {self.record_step}')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.savefig('loss_over_interval.png')

        accelerator.print('training complete')
        writer.close()
