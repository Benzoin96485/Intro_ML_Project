#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Weiliang Luo
    Student No.: 1900011804
    Date changed: 2021.4.17
    Python Version: Anaconda3 (Python 3.8.6)
'''

# Imports
# std libs

# 3rd Party libs
import torch
import torch.nn.functional as F

# my modules
from parse import parseArgs
import small_cnn as scnn

class Env():
    def __init__(self):
        self.FLAGS = parseArgs()
        self.isGpu = torch.cuda.is_available()
        self.cp_path = './MNIST_small_cnn.checkpoint'
        self.nn = scnn.create_network()
        self.epsilons = [0, .05, .1, .15, .2, .25, .3]
        self.loss_func_fgsm = F.nll_loss
        self.loss_func_pgd = F.nll_loss
        self.iter_pgd = 10

    def toGPU(self):
        if self.isGpu:
            print('gpu used\n')
            device = torch.device('cuda:0')
            model = self.nn.to(device)
        else:
            print('gpu is not available, try to use cpu\n')
            device = None
            model = self.nn
        return model, device