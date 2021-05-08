#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Weiliang Luo
    Student No.: 1900011804
    Date changed: 2021.5.2
    Python Version: Anaconda3 (Python 3.8.6)
'''
# Imports
# std libs

# 3rd Party libs
from small_cnn import test
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# my modules

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

def load_data():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    return test_loader

class Fgsm():
    def __init__(self, model, device, loader, env):
        self.m = model
        self.d = device
        self.l = loader
        self.e = env

    def fgsm(self, image, epsilon, grad, o_image=None):
        new_image = image + epsilon * grad.sign() # 对图像进行扰动
        new_image = torch.clamp(new_image, 0, 1) # 截断到 [0,1] 区间内
        return new_image

    def fgsm_attack(self, epsilon):
        loss_func = self.e.loss_func_fgsm
        total = len(self.l)
        adv_exp = []
        crt = 0

        for data, label in self.l:
            data, label = data.to(self.d), label.to(self.d)
            if self.e.FLAGS.randinit == True:
                datas = []
                for i in range(10):
                    datas.append(data + torch.Tensor(np.random.uniform(-epsilon, epsilon, data.shape)).type_as(data).to(self.d))
            else:
                datas = [data]
            token = False
            for target in datas:
                target.requires_grad = True
                out = self.m(target)
                pred = out.max(1, keepdim=True)[1]
                if pred.item() != label.item():
                    # 原模型已经分类错误，无需攻击
                    token = True
                    break
                self.m.zero_grad()
                loss = loss_func(out, label)
                loss.backward() # 计算前馈过程中损失函数在模型上的梯度
                grad = target.grad.data
                new_data = self.fgsm(target, epsilon, grad) # 进行攻击
                new_data = data + torch.clamp(new_data - data, -epsilon, epsilon)
                out = self.m(new_data) # 重新分类
                new_pred = out.max(1, keepdim=True)[1]
                if new_pred != label:
                    token = True
                    if len(adv_exp) < 5:
                        exp = new_data.squeeze().detach().cpu().numpy()
                        adv_exp.append((pred.item(), new_pred.item(), exp))
                    break
                else:
                    if len(adv_exp) < 5 and epsilon == 0:
                        exp = new_data.squeeze().detach().cpu().numpy()
                        adv_exp.append((pred.item(), new_pred.item(), exp))

            if token:
                continue
            else:
                crt += 1
            
        acc = crt / total
        suc = 1 - acc
        print("Epsilon: {}; Accuracy: {}; Success rate: {:.4}".format(epsilon, acc, suc))
        return acc, adv_exp

    def draw(self, examples):
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
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.show()        

    def eval_fgsm(self):
        examples = []
        for epsilon in self.e.epsilons:
            acc, ex = self.fgsm_attack(epsilon)
            examples.append(ex)
        if self.e.FLAGS.draw:
            self.draw(examples)     


class Pgd():
    def __init__(self, model, device, loader, env):
        self.m = model
        self.d = device
        self.l = loader
        self.e = env

    def pgd_step(self, image, pert, label, epsilon, eps_iter):
        new_image = image + pert
        new_image = Variable(new_image)
        new_image.requires_grad = True
        loss_func = self.e.loss_func_pgd
        out = self.m(new_image)
        loss = loss_func(out, label)
        self.m.zero_grad()
        loss.backward()
        grad = new_image.grad.data
        new_image = new_image.detach()
        pert += eps_iter * grad.sign()
        pert = torch.clamp(pert, -epsilon, epsilon)
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
                pert = torch.Tensor(np.random.uniform(-epsilon, epsilon, data.shape)).type_as(data).to(self.d)
            else:
                pert = torch.zeros(data.shape).type_as(data).to(self.d)
            for i in range(self.e.FLAGS.pgditer):
                pert = self.pgd_step(data, pert, label, epsilon ,eps_iter) # 进行单步迭代攻击
            new_data = data + pert
            new_data = torch.clamp(new_data, 0, 1)
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
    def __init__(self, model, device, loader, env):
        self.m = model
        self.d = device
        self.l = loader
        self.e = env

    def get_probs(self, data, label):
        out = self.m(data.to(self.d)).cpu()
        probs = nn.Softmax()(out)[:, label]
        return torch.diag(probs.data)
    
    def simba(self, data, label, epsilon, num_iters=500):
        n_dim = data.view(-1, 1).size()[0]
        perm = torch.randperm(n_dim)
        last_prob = self.get_probs(data, label)
        for i in range(num_iters):
            diff = torch.zeros(n_dim)
            diff[perm[i]] = epsilon
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