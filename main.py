
import argparse
import math
import random
import os
import functools
import operator
import contextlib
import warnings
import numpy as np
import lmdb
from PIL import Image
from tqdm import tqdm
from io import BytesIO

enabled = True
weight_gradients_disabled = False

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from torch import autograd, optim
from torch.utils import data
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms, utils

from utility import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from utility import reduce_sum

##  Some golbal parameters declaration..

p_augment = 0  
target_aug = 0.4  ## setting the setting of target augmentation.
multi_folds_channel = 1 # (may change it to 2, on basis of results)
size_of_image = 32 
learning_rate = 0.002
R1_WEIGHT = 10    #  weight of the r1 regularization
NUMBER_OF_SAMPLE = 64    # number of the samples generated during training
MIXING = 0.9   # probability of latent code mixing
AL = 500 * 1000    #  target duraing to reach augmentation probability for adaptive augmentation
R1_REG_INT = 16    #  interval of the applying r1 regularization
PL_REG = 4   #  interval of the applying path length regularization
PBS = 2    #  batch size reducing factor for the path length regularization (reduce memory consumption)
PATH_REGULARISATION = 2   #  weight of the path length regularization

####  A class to open the lmdb file and return a image (designed for space preservation).

try:
    import wandb

except ImportError:
    wandb = None

class DRM(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

###  Artificial Augmenetation part (Remove them on basis of computational requirements 
###  and accuracy achieved by model

class AdaptiveAugment:
    def __init__(self, ada_aug_target, ada_aug_len, update_every, device):
        self.ada_aug_target = ada_aug_target
        self.ada_aug_len = ada_aug_len
        self.update_every = update_every

        self.ada_update = 0
        self.ada_aug_buf = torch.tensor([0.0, 0.0], device=device)
        self.r_t_stat = 0
        self.ada_aug_p = 0

    @torch.no_grad()
    def tune(self, real_pred):
        self.ada_aug_buf += torch.tensor(
            (torch.sign(real_pred).sum().item(), real_pred.shape[0]),
            device=real_pred.device,
        )
        self.ada_update += 1

        if self.ada_update % self.update_every == 0:
            self.ada_aug_buf = reduce_sum(self.ada_aug_buf)
            pred_signs, n_pred = self.ada_aug_buf.tolist()

            self.r_t_stat = pred_signs / n_pred

            if self.r_t_stat > self.ada_aug_target:
                sign = 1

            else:
                sign = -1

            self.ada_aug_p += sign * n_pred / self.ada_aug_len
            self.ada_aug_p = min(1, max(0, self.ada_aug_p))
            self.ada_aug_buf.mul_(0)
            self.ada_update = 0

        return self.ada_aug_p


augmentation_arg = (
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148,
)


def translate_mat(t_x, t_y, device="cpu"):
    batch = t_x.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y), 1)
    mat[:, :2, 2] = translate

    return mat


def rotate_mat(theta, device="cpu"):
    batch = theta.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    rot = torch.stack((cos_t, -sin_t, sin_t, cos_t), 1).view(batch, 2, 2)
    mat[:, :2, :2] = rot

    return mat


def scale_mat(s_x, s_y, device="cpu"):
    batch = s_x.shape[0]

    mat = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y

    return mat


def translate3d_mat(t_x, t_y, t_z):
    batch = t_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    translate = torch.stack((t_x, t_y, t_z), 1)
    mat[:, :3, 3] = translate

    return mat


def rotate3d_mat(axis, theta):
    batch = theta.shape[0]

    u_x, u_y, u_z = axis

    eye = torch.eye(3).unsqueeze(0)
    cross = torch.tensor([(0, -u_z, u_y), (u_z, 0, -u_x), (-u_y, u_x, 0)]).unsqueeze(0)
    outer = torch.tensor(axis)
    outer = (outer.unsqueeze(1) * outer).unsqueeze(0)

    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    rot = cos_t * eye + sin_t * cross + (1 - cos_t) * outer

    eye_4 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    eye_4[:, :3, :3] = rot

    return eye_4


def scale3d_mat(s_x, s_y, s_z):
    batch = s_x.shape[0]

    mat = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, 0, 0] = s_x
    mat[:, 1, 1] = s_y
    mat[:, 2, 2] = s_z

    return mat


def luma_flip_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    flip = 2 * torch.ger(axis, axis) * i.view(-1, 1, 1)

    return eye - flip


def saturation_mat(axis, i):
    batch = i.shape[0]

    eye = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    axis = torch.tensor(axis + (0,))
    axis = torch.ger(axis, axis)
    saturate = axis + (eye - axis) * i.view(-1, 1, 1)

    return saturate


def lognormal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).log_normal_(mean=mean, std=std)


def category_sample(size, categories, device="cpu"):
    category = torch.tensor(categories, device=device)
    sample = torch.randint(high=len(categories), size=(size,), device=device)

    return category[sample]


def uniform_sample(size, low, high, device="cpu"):
    return torch.empty(size, device=device).uniform_(low, high)


def normal_sample(size, mean=0, std=1, device="cpu"):
    return torch.empty(size, device=device).normal_(mean, std)


def bernoulli_sample(size, p, device="cpu"):
    return torch.empty(size, device=device).bernoulli_(p)


def random_mat_apply(p, transform, prev, eye, device="cpu"):
    size = transform.shape[0]
    select = bernoulli_sample(size, p, device=device).view(size, 1, 1)
    select_transform = select * transform + (1 - select) * eye

    return select_transform @ prev


def sample_affine(p, size, height, width, device="cpu"):
    G = torch.eye(3, device=device).unsqueeze(0).repeat(size, 1, 1)
    eye = G

    # flip
    param = category_sample(size, (0, 1))
    Gc = scale_mat(1 - 2.0 * param, torch.ones(size), device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('flip', G, scale_mat(1 - 2.0 * param, torch.ones(size)), sep='\n')

    # 90 rotate
    param = category_sample(size, (0, 3))
    Gc = rotate_mat(-math.pi / 2 * param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('90 rotate', G, rotate_mat(-math.pi / 2 * param), sep='\n')

    # integer translate
    param = uniform_sample(size, -0.125, 0.125)
    param_height = torch.round(param * height) / height
    param_width = torch.round(param * width) / width
    Gc = translate_mat(param_width, param_height, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('integer translate', G, translate_mat(param_width, param_height), sep='\n')

    # isotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('isotropic scale', G, scale_mat(param, param), sep='\n')

    p_rot = 1 - math.sqrt(1 - p)

    # pre-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param, device=device)
    G = random_mat_apply(p_rot, Gc, G, eye, device=device)
    # print('pre-rotate', G, rotate_mat(-param), sep='\n')

    # anisotropic scale
    param = lognormal_sample(size, std=0.2 * math.log(2))
    Gc = scale_mat(param, 1 / param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('anisotropic scale', G, scale_mat(param, 1 / param), sep='\n')

    # post-rotate
    param = uniform_sample(size, -math.pi, math.pi)
    Gc = rotate_mat(-param, device=device)
    G = random_mat_apply(p_rot, Gc, G, eye, device=device)
    # print('post-rotate', G, rotate_mat(-param), sep='\n')

    # fractional translate
    param = normal_sample(size, std=0.125)
    Gc = translate_mat(param, param, device=device)
    G = random_mat_apply(p, Gc, G, eye, device=device)
    # print('fractional translate', G, translate_mat(param, param), sep='\n')

    return G


def sample_color(p, size):
    C = torch.eye(4).unsqueeze(0).repeat(size, 1, 1)
    eye = C
    axis_val = 1 / math.sqrt(3)
    axis = (axis_val, axis_val, axis_val)

    # brightness
    param = normal_sample(size, std=0.2)
    Cc = translate3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # contrast
    param = lognormal_sample(size, std=0.5 * math.log(2))
    Cc = scale3d_mat(param, param, param)
    C = random_mat_apply(p, Cc, C, eye)

    # luma flip
    param = category_sample(size, (0, 1))
    Cc = luma_flip_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # hue rotation
    param = uniform_sample(size, -math.pi, math.pi)
    Cc = rotate3d_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    # saturation
    param = lognormal_sample(size, std=1 * math.log(2))
    Cc = saturation_mat(axis, param)
    C = random_mat_apply(p, Cc, C, eye)

    return C


def make_grid(shape, x0, x1, y0, y1, device):
    n, c, h, w = shape
    grid = torch.empty(n, h, w, 3, device=device)
    grid[:, :, :, 0] = torch.linspace(x0, x1, w, device=device)
    grid[:, :, :, 1] = torch.linspace(y0, y1, h, device=device).unsqueeze(-1)
    grid[:, :, :, 2] = 1

    return grid


def affine_grid(grid, mat):
    n, h, w, _ = grid.shape
    return (grid.view(n, h * w, 3) @ mat.transpose(1, 2)).view(n, h, w, 2)


def get_padding(G, height, width, kernel_size):
    device = G.device

    cx = (width - 1) / 2
    cy = (height - 1) / 2
    cp = torch.tensor(
        [(-cx, -cy, 1), (cx, -cy, 1), (cx, cy, 1), (-cx, cy, 1)], device=device
    )
    cp = G @ cp.T

    pad_k = kernel_size // 4

    pad = cp[:, :2, :].permute(1, 0, 2).flatten(1)
    pad = torch.cat((-pad, pad)).max(1).values
    pad = pad + torch.tensor([pad_k * 2 - cx, pad_k * 2 - cy] * 2, device=device)
    pad = pad.max(torch.tensor([0, 0] * 2, device=device))
    pad = pad.min(torch.tensor([width - 1, height - 1] * 2, device=device))

    pad_x1, pad_y1, pad_x2, pad_y2 = pad.ceil().to(torch.int32)

    return pad_x1, pad_x2, pad_y1, pad_y2


def try_sample_affine_and_pad(img, p, kernel_size, G=None):
    batch, _, height, width = img.shape

    G_try = G

    if G is None:
        G_try = torch.inverse(sample_affine(p, batch, height, width))

    pad_x1, pad_x2, pad_y1, pad_y2 = get_padding(G_try, height, width, kernel_size)

    img_pad = F.pad(img, (pad_x1, pad_x2, pad_y1, pad_y2), mode="reflect")

    return img_pad, G_try, (pad_x1, pad_x2, pad_y1, pad_y2)


class GridSampleForward(autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        out = F.grid_sample(
            input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        ctx.save_for_backward(input, grid)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = GridSampleBackward.apply(grad_output, input, grid)

        return grad_input, grad_grid


class GridSampleBackward(autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")
        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_grid):
        grid, = ctx.saved_tensors
        grad_grad_output = None

        if ctx.needs_input_grad[0]:
            grad_grad_output = GridSampleForward.apply(grad_grad_input, grid)

        return grad_grad_output, None, None


grid_sample = GridSampleForward.apply


def scale_mat_single(s_x, s_y):
    return torch.tensor(((s_x, 0, 0), (0, s_y, 0), (0, 0, 1)), dtype=torch.float32)


def translate_mat_single(t_x, t_y):
    return torch.tensor(((1, 0, t_x), (0, 1, t_y), (0, 0, 1)), dtype=torch.float32)


def random_apply_affine(img, p, G=None, antialiasing_kernel=augmentation_arg
):
    kernel = antialiasing_kernel
    len_k = len(kernel)

    kernel = torch.as_tensor(kernel).to(img)
    # kernel = torch.ger(kernel, kernel).to(img)
    kernel_flip = torch.flip(kernel, (0,))

    img_pad, G, (pad_x1, pad_x2, pad_y1, pad_y2) = try_sample_affine_and_pad(
        img, p, len_k, G
    )

    G_inv = (
        translate_mat_single((pad_x1 - pad_x2).item() / 2, (pad_y1 - pad_y2).item() / 2)
        @ G
    )
    up_pad = (
        (len_k + 2 - 1) // 2,
        (len_k - 2) // 2,
        (len_k + 2 - 1) // 2,
        (len_k - 2) // 2,
    )
    img_2x = upfirdn2d(img_pad, kernel.unsqueeze(0), up=(2, 1), pad=(*up_pad[:2], 0, 0))
    img_2x = upfirdn2d(img_2x, kernel.unsqueeze(1), up=(1, 2), pad=(0, 0, *up_pad[2:]))
    G_inv = scale_mat_single(2, 2) @ G_inv @ scale_mat_single(1 / 2, 1 / 2)
    G_inv = translate_mat_single(-0.5, -0.5) @ G_inv @ translate_mat_single(0.5, 0.5)
    batch_size, channel, height, width = img.shape
    pad_k = len_k // 4
    shape = (batch_size, channel, (height + pad_k * 2) * 2, (width + pad_k * 2) * 2)
    G_inv = (
        scale_mat_single(2 / img_2x.shape[3], 2 / img_2x.shape[2])
        @ G_inv
        @ scale_mat_single(1 / (2 / shape[3]), 1 / (2 / shape[2]))
    )
    grid = F.affine_grid(G_inv[:, :2, :].to(img_2x), shape, align_corners=False)
    img_affine = grid_sample(img_2x, grid)
    d_p = -pad_k * 2
    down_pad = (
        d_p + (len_k - 2 + 1) // 2,
        d_p + (len_k - 2) // 2,
        d_p + (len_k - 2 + 1) // 2,
        d_p + (len_k - 2) // 2,
    )
    img_down = upfirdn2d(
        img_affine, kernel_flip.unsqueeze(0), down=(2, 1), pad=(*down_pad[:2], 0, 0)
    )
    img_down = upfirdn2d(
        img_down, kernel_flip.unsqueeze(1), down=(1, 2), pad=(0, 0, *down_pad[2:])
    )

    return img_down, G


def apply_color(img, mat):
    batch = img.shape[0]
    img = img.permute(0, 2, 3, 1)
    mat_mul = mat[:, :3, :3].transpose(1, 2).view(batch, 1, 3, 3)
    mat_add = mat[:, :3, 3].view(batch, 1, 1, 3)
    img = img @ mat_mul + mat_add
    img = img.permute(0, 3, 1, 2)

    return img


def random_apply_color(img, p, C=None):
    if C is None:
        C = sample_color(p, img.shape[0])

    img = apply_color(img, C.to(img))

    return img, C


def augment(img, p, transform_matrix=(None, None)):
    img, G = random_apply_affine(img, p, transform_matrix[0])
    img, C = random_apply_color(img, p, transform_matrix[1])

    return img, (G, C)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

## defining the function which takes care of training part..

def SG2_training(iterations, batch_size, loader, generator, discriminator, g_optim, d_optim, g_ema, device):

    augment=True
    WDB = True
    LATENT = 512
    START_ITER = 0

    loader = sample_data(loader)

    pbar = range(iterations)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=START_ITER, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = p_augment if p_augment > 0 else 0.0
    r_t_stat = 0

    if augment and p_augment == 0:
        ada_augment = AdaptiveAugment(target_aug, AL, 8, device)

    sample_z = torch.randn(NUMBER_OF_SAMPLE, LATENT, device=device)

    for idx in pbar:
        i = idx + START_ITER

        if i > iterations:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(batch_size, LATENT, MIXING, device)
        fake_img, _ = generator(noise)

        if augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if augment and p_augment== 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % R1_REG_INT == 0

        if d_regularize:
            real_img.requires_grad = True

            if augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (R1_WIGHT / 2 * r1_loss * R1_REG_INT + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(batch_size, LATENT, MIXING, device)
        fake_img, _ = generator(noise)

        if augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % PL_REG == 0

        if g_regularize:
            path_batch_size = max(1, batch_size // PBS)
            noise = mixing_noise(path_batch_size, LATENT, MIXING, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = PATH_REGULARISATION * PL_REG * path_loss

            if PBS:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and WDB:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(NUMBER_OF_SAMPLE ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )

####  The main execution portion of arguments intake, calling and running the networks.
 
device = "cuda"

def training(path, iterations, batch_size, checkpoint, augment=True, wandb=True):

    WDB = True
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    LATENT = 512
    NMLP = 8
    START_ITER = 0

    ###  Calling the Generator network and Descriminator Network.

    from model import Generator, Discriminator

    generator = Generator(
        size_of_image, LATENT, NMLP, channel_multiplier=multi_folds_channel
    ).to(device)
    discriminator = Discriminator(
        size_of_image, channel_multiplier=multi_folds_channel
    ).to(device)
    g_ema = Generator(
        size_of_image, LATENT, NMLP, channel_multiplier=multi_folds_channel
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    ###  Optimisers selection and hyperparameters selection.

    g_reg_ratio = PL_REG / (PL_REG + 1)
    d_reg_ratio = R1_REG_INT / (R1_REG_INT + 1)

    ###   Testing Adam :

    g_optim = optim.Adam(
        generator.parameters(),
        lr=learning_rate * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    ###   Testing RMS Prop : 

    # g_optim = optim.RMSProp(
    #     generator.parameters(),
    #     lr=learning_rate * g_reg_ratio,
    #     betas=(0.99 ** g_reg_ratio),
    # )
    # d_optim = optim.RMSProp(
    #     discriminator.parameters(),
    #     lr=learning_rate * d_reg_ratio,
    #     alpha=(0.99 ** d_reg_ratio),
    # )

    if checkpoint is not None:
        print("load model:", checkpoint)

        ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(checkpoint)
            START_ITER = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = DRM(path, transform, size_of_image)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and WDB:
        wandb.init(project="stylegan 2")

    SG2_training(iterations, batch_size, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
