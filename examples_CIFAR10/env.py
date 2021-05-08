#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Weiliang Luo
    Student No.: 1900011804
    Date changed: 2021.5.8
    Python Version: Anaconda3 (Python 3.8.6)
    本文件用于设置实验的基本环境，整个实验请用支持 CUDA 的 GPU 运行
'''

# Imports
# std libs

# 3rd Party libs
import torch
import torch.nn.functional as F

# my modules
from parse import parseArgs
import PreActResNet18 as res

class Env():
    def __init__(self):
        self.FLAGS = parseArgs() # 命令行参数
        self.isGpu = torch.cuda.is_available() # 判断 GPU 是否可用
        self.cp_path = './CIFAR10_PreActResNet18.checkpoint' # CIFAR10 预训练参数路径
        self.nn = res.PreActResNet18() # 初始化网络
        #self.epsilons = [0, .03]
        self.epsilons = [0, .005, .01, .015, .02, .025, .03] # 扰动限制
        self.loss_func_fgsm = F.cross_entropy # fgsm 损失函数
        self.loss_func_pgd = F.nll_loss # pgd 损失函数
        self.iter_pgd = 10 # pgd 循环次数
        self.mapping = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def toGPU(self):
        # 将设备设为当前可用的第 0 个 CUDA 设备，并将模型转到 GPU 上计算
        if self.isGpu:
            print('gpu used\n')
            device = torch.device('cuda:0')
            model = self.nn.to(device)
        else:
            print('gpu is not available, try to use cpu\n')
            device = None
            model = self.nn
        return model, device