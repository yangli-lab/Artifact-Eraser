import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from glob import glob
from PIL import Image

from dataset.DataSet.dataset_celebdf import CELEB_DF
from dataset.DataSet.dataset_ffpp import FFPP
from dataset.DataSet.dataset_dfdc import DFDC
from data_transform.transform_config import get_transform
from utils import common, train_utils
from encodecode import EncodeDecode
from criteria.lpips.lpips import LPIPS
from criteria import id_loss, moco_loss
from training.ranger import Ranger
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger

random.seed(0)
torch.manual_seed(0)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class Train():
    def __init__(self, opts, prev_train_checkpoint = None):# opts.lpips_type, opts.lpips_lambda, opts.id_lambda, opts.exp_dir
        self.opts = opts                                    #opts.save_interval, opts.max_steps
        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device
        #TODO: initialize network
        self.model = EncodeDecode(opts).to(self.device)

        # initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type = self.opts.lpips_type).to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()

        self.mse_loss = nn.MSELoss().to(self.device).eval()

        # initialize optimizer
        self.optimizer = self.configure_optimizers()

        # initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        #initialize dataset
        #TODO:
        # only train, no test
        transform_dict = get_transform()
        # dataset = CELEB_DF(root_dir = '',
        #         mix_real_fake = False,
        #         transform = transform_dict['gt_train'],
        #         train_target = 'e4e', train_test = 'gt_train')
        # dataset = FFPP(root_dir = '',
        #                  mix_real_fake = False, transform = transform_dict['gt_train'], 
        #             train_target = 'e4e', train_test = 'gt_train')
        dataset = DFDC(root_dir = '',
                    label_file = '',
                    transform = transform_dict['gt_train'], 
                    train_target = 'e4e', train_test = 'gt_train')
        self.train_dataloader = DataLoader(dataset, 
                                        batch_size = self.opts.batch_size,
                                        shuffle = True,
                                        num_workers = int(self.opts.workers),
                                        drop_last = True
                                )

        print(dataset.__len__())
        #logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok = True)
        self.logger = SummaryWriter(log_dir = log_dir)

        #initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    
    def load_from_midtrain_checkpoint(self, ckpt):# opts.keep_optimizer
        print('load_previous training checkpoint...')
        #self.global_step = ckpt['global_step'] + 1
        self.model.load_state_dict([ckpt['state_dict']])
        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Resuming training from step {self.global_step}')

    
    def configure_optimizers(self):# opts.train_decoder, opts.learning_rate
        params = list(self.model.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.model.decoder.parameters())
        else:
            self.requires_grad(self.model.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr = self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr = self.opts.learning_rate)
        return optimizer
    

    def train(self):
        self.model.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                x, y, y_hat, latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y,y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if self.global_step != 0:
                        self.checkpoint_me(loss_dict, is_best=False)
                # finish training
                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'interation_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def forward(self, batch):
        x, y = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        y_hat, latent = self.model.forward(x, return_latents=True)
        if self.opts.dataset_type == "cars_encode":
            y_hat = y_hat[:, :, 32:224, :]
        return x, y, y_hat, latent

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.model.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and self.model.encoder.progressive_stage.value != 18:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.model.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.model.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs


    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.model.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.model.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['loss_now'] = self.loss_dict
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.model.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.model.encoder.progressive_stage.value + 1]


    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.model.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.model.encoder.set_progressive_stage(ProgressiveStage(i))


    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    def discriminator_loss(self, real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss


    def discriminator_r1_loss(self, real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag


    def train_discriminator(self, batch):
        loss_dict = {}
        x, _ = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.model.decoder.get_latent(sample_z)
        fake_w = self.model.encoder(x)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.model.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w