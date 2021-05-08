#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Weiliang Luo
    Student No.: 1900011804
    Date changed: 2021.5.8
    Python Version: Anaconda3 (Python 3.8.6)
'''
# Imports
# std libs

# 3rd Party libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# my modules

def load_model(model, path):
    checkpoint = torch.load(path) # 读取预训练参数
    model.load_state_dict(checkpoint['state_dict']) # 将预训练参数加载到模型
    model.eval() # 将模型转换到预测（评估）模式

def load_data():
    # 加载 pytorch 自带的 CIFAR10 数据集
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    # 注意：由于攻击时一张一张进行，batchsize 要设为 1，且预训练没有进行任何图像数据增强和预处理，transform 只需要 totensor
    return test_loader

class Fgsm():
    # FGSM （快速梯度符号法） 攻击
    # 论文：https://arxiv.org/abs/1412.6572
    def __init__(self, model, device, loader, env):
        self.m = model # 加载好预训练参数的模型
        self.d = device # GPU
        self.l = loader # 数据集加载器
        self.e = env # 环境类

    def fgsm(self, image, epsilon, grad, o_image=None):
        new_image = image + epsilon * grad.sign() # 对图像进行扰动
        new_image = torch.clamp(new_image, 0, 1) # 截断到 [0,1] 区间内
        return new_image

    def fgsm_attack(self, epsilon):
        loss_func = self.e.loss_func_fgsm # 损失函数
        total = len(self.l) # 数据集容量
        adv_exp = [] # 对抗样本的例子
        crt = 0 # 识别正确率

        for data, label in self.l:
            data, label = data.to(self.d), label.to(self.d) # 数据集转到 GPU
            '''
            if self.e.FLAGS.randinit == True:
                datas = []
                for i in range(10):
                    datas.append(data + torch.Tensor(np.random.uniform(-epsilon, epsilon, data.shape)).type_as(data).to(self.d))
            else:
                datas = [data]
            这是一个失败的尝试，原本想引入随机初始化策略，即刚开始就对同一个样本尝试随机扰动十次，得到十个攻击对象，在此基础上
            进行 FGSM，如果有一个攻击成功（图像分类失败）就算攻击成功，后来这个尝试效果很差，故放弃
            '''
            datas = [data] # 攻击样本
            token = False # 可以忽略
            for target in datas:
                # 失败尝试遗留物，实际上不需要这个循环
                target.requires_grad = True # 告诉 torch 要获取被攻击样本接下来的运算的梯度
                out = self.m(target) # 模型在样本原图上的输出概率向量
                pred = out.max(1, keepdim=True)[1] # 模型在样本原图上的预测结果
                if pred.item() != label.item():
                    # 原模型已经分类错误，无需攻击
                    token = True
                    break
                self.m.zero_grad() # 将模型上的梯度归零
                loss = loss_func(out, label) # 计算损失函数
                loss.backward() # 计算前馈过程中损失函数在模型上的梯度
                grad = target.grad.data # 获取梯度
                new_data = self.fgsm(target, epsilon, grad) # 基于梯度进行攻击
                new_data = data + torch.clamp(new_data - data, -epsilon, epsilon) # 将扰动限制在 epsilon 范围内
                out = self.m(new_data) # 传入对抗样本重新分类
                new_pred = out.max(1, keepdim=True)[1] # 获得分类结果
                if new_pred != label:
                    token = True
                    if len(adv_exp) < 5:
                        # 存储一些对抗样本的例子，可视化用
                        exp = new_data.squeeze().detach().cpu().numpy()
                        adv_exp.append((pred.item(), new_pred.item(), exp))
                    break
                else:
                    if len(adv_exp) < 5 and epsilon == 0:
                        # 存储一些原图的例子，可视化用
                        exp = new_data.squeeze().detach().cpu().numpy()
                        adv_exp.append((pred.item(), new_pred.item(), exp))

            if token:
                continue
            else:
                crt += 1 # 如果攻击后依然识别正确，识别正确数+1
            
        acc = crt / total # 计算对抗样本的识别正确率
        suc = 1 - acc # 计算攻击成功率
        print("Epsilon: {}; Accuracy: {}; Success rate: {:.4}".format(epsilon, acc, suc))
        return acc, adv_exp

    def draw(self, examples):
        # 绘图函数，展示原图和不同规模扰动下的对抗样本
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(self.e.epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(self.e.epsilons),len(examples[0]),cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(self.e.epsilons[i]), fontsize=14)
                orig,adv,ex = examples[i][j]
                plt.title("{} -> {}".format(self.e.mapping[orig], self.e.mapping[adv]))
                plt.imshow(ex.transpose(1, 2, 0))
        plt.tight_layout()
        plt.show()        

    def eval_fgsm(self):
        # 对 env 中不同的 epsilon 设置进行分别尝试，评估攻击结果
        examples = []
        for epsilon in self.e.epsilons:
            acc, ex = self.fgsm_attack(epsilon)
            examples.append(ex)
        if self.e.FLAGS.draw:
            self.draw(examples)     


class Pgd():
    # PGD （投影梯度下降） 攻击
    # 论文：https://arxiv.org/abs/1706.06083
    def __init__(self, model, device, loader, env):
        self.m = model
        self.d = device
        self.l = loader
        self.e = env

    def pgd_step(self, image, pert, label, epsilon, eps_iter):
        new_image = image + pert # 添加上一次迭代产生的扰动
        new_image = Variable(new_image) # 将图像变为具有梯度的 Variable 变量
        new_image.requires_grad = True # 要求获取梯度
        loss_func = self.e.loss_func_pgd # 加载损失函数
        out = self.m(new_image) # 将图像送入神经网络
        loss = loss_func(out, label) # 计算损失函数
        self.m.zero_grad() # 清空计算产生的梯度
        loss.backward() # 反向传播所得梯度
        grad = new_image.grad.data # 获取梯度
        new_image = new_image.detach() # 从梯度模式中脱离
        pert += eps_iter * grad.sign() # 在原有扰动上叠加新的一步
        pert = torch.clamp(pert, -epsilon, epsilon) # 始终保证返回的扰动在限制范围内
        return pert

    def pgd_attack(self, epsilon, eps_iter):
        total = len(self.l)
        crt = 0

        for data, label in self.l:
            data, label = data.to(self.d), label.to(self.d)
            out = self.m(data)
            pred = out.max(1, keepdim=True)[1]
            if pred.item() != label.item(): # 原模型已经分类错误，无需攻击
                continue
            self.m.zero_grad()
            if self.e.FLAGS.randinit:
                # PGD 的随机初始化策略
                pert = torch.Tensor(np.random.uniform(-epsilon, epsilon, data.shape)).type_as(data).to(self.d)
            else:
                pert = torch.zeros(data.shape).type_as(data).to(self.d)
            for i in range(self.e.FLAGS.pgditer):
                pert = self.pgd_step(data, pert, label, epsilon ,eps_iter) # 进行单步迭代攻击
            new_data = data + pert
            new_data = torch.clamp(new_data, 0, 1) # 截断到图像区间内
            out = self.m(new_data) # 重新分类
            new_pred = out.max(1, keepdim=True)[1]
            if new_pred == label:
                crt += 1

        acc = crt / total
        suc = 1 - acc
        print("Epsilon: {}; Stepsize: {}; Accuracy: {}; Success rate: {:.4}".format(epsilon, eps_iter, acc, suc))
        return acc           

    def eval_pgd(self):
        for epsilon in self.e.epsilons[1:]:
            stepsize = [epsilon, epsilon/2, epsilon/4, epsilon/8] # 步长
            for eps_iter in stepsize:
                self.pgd_attack(epsilon, eps_iter)


class Simba():
    # SIMBA（简单黑盒攻击）
    # 论文：https://arxiv.org/abs/1905.07121
    def __init__(self, model, device, loader, env):
        self.m = model
        self.d = device
        self.l = loader
        self.e = env

    def get_probs(self, data, label):
        out = self.m(data.to(self.d)).cpu()
        probs = nn.Softmax()(out)[:, label] # 获得神经网络输出的概率向量
        return torch.diag(probs.data)
    
    def simba(self, data, label, epsilon, num_iters=500):
        n_dim = data.view(-1, 1).size()[0]
        perm = torch.randperm(n_dim) # 获得随机乱序排列，用于决定接下来扰动哪些像素
        last_prob = self.get_probs(data, label) # 原图对应的概率输出
        for i in range(num_iters):
            diff = torch.zeros(n_dim)
            diff[perm[i]] = epsilon # 产生一个单像素扰动
            new_prob = self.get_probs((data - diff.view(data.size())).clamp(0, 1), label)
            if new_prob < last_prob:
                data = (data - diff.view(data.size())).clamp(0, 1)
                last_prob = new_prob
            else:
                new_prob = self.get_probs((data + diff.view(data.size())).clamp(0, 1), label)
                if new_prob < last_prob:
                    data = (data + diff.view(data.size())).clamp(0, 1)
                    last_prob = new_prob
        return data
    
    def simba_attack(self, epsilon, num_iters, amount=1000):
        total = len(self.l)
        crt = 0
        i = 0

        for data, label in self.l:
            i += 1
            new_data = self.simba(data, label, epsilon, num_iters).to(self.d)
            label = label.to(self.d)
            out = self.m(new_data) # 重新分类
            new_pred = out.max(1, keepdim=True)[1]
            if new_pred == label:
                crt += 1
            #if i % 10 == 0:
                #print("{} pcs and {:.4} ratio".format(i, crt/i))
            if i == amount:
                break

        acc = crt / amount
        suc = 1 - acc
        print("Epsilon: {}; num_of_iters: {}; Accuracy: {}; Success rate: {:.4}".format(epsilon, num_iters, acc, suc))
        return acc 

    def eval_simba(self):
        for epsilon in self.e.epsilons[1:]:
            for num_iters in [100, 300, 500, 700]:
                self.simba_attack(epsilon, num_iters)