import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import kornia.augmentation as A
import os
import csv
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

term_width = 80

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time

def get_transform(opt):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.247, 0.243, 0.261])

    elif opt.dataset == "mnist":
        mean = torch.tensor([0.5])
        std = torch.tensor([0.5])

    transforms_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transforms_list)


def Denormalizer(opt):
    if opt.dataset == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.247, 0.243, 0.261])
    elif opt.dataset == "mnist":
        mean = torch.tensor([0.5])
        std = torch.tensor([0.5])

    de_mean = -mean / std
    de_std = 1 / std
    denormalizer = transforms.Normalize(de_mean, de_std)
    return denormalizer

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_dataloader(opt, train):
    transform = get_transform(opt)
    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, shuffle=True)  #删掉num_workers
    return dataloader


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()