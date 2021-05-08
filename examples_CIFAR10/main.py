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



# my modules
import prp
from env import Env

def main():
    env = Env() # 初始化环境
    model, device = env.toGPU() # 转到 GPU 上运行
    prp.load_model(model, env.cp_path) # 加载预训练参数
    loader = prp.load_data() # 加载数据集

    if env.FLAGS.fgsm:
        print("FGSM attack running\n -----")
        attack = prp.Fgsm(model, device, loader, env)
        attack.eval_fgsm()

    if env.FLAGS.pgd:
        print("PGD attack running\n -----")
        attack = prp.Pgd(model, device, loader, env)
        attack.eval_pgd()

    if env.FLAGS.black:
        print("black box attack running\n -----")
        attack = prp.Simba(model, device, loader, env)
        attack.eval_simba()

if __name__ == '__main__':
    main()