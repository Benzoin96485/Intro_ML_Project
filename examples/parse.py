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
import argparse

# 3rd Party libs

# my modules

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=bool,
        default=True,
        help="try to use cuda"
    )
    parser.add_argument(
        '--fgsm',
        type=bool,
        default=False,
        help="run fgsm attack"
    )
    parser.add_argument(
        '--draw',
        type=bool,
        default=False,
        help="draw fgsm adversial examples"
    )
    parser.add_argument(
        '--pgd',
        type=bool,
        default=False,
        help="run pgd attack"
    )
    parser.add_argument(
        '--black',
        type=bool,
        default=False,
        help="run blackbox attack"
    )
    parser.add_argument(
        '--randinit',
        type=bool,
        default=False,
        help="random initialize in pgd"
    )
    parser.add_argument(
        '--pgditer',
        type=int,
        default=10,
        help="iteration number in pgd"
    )
    flags, unparsed = parser.parse_known_args()
    return flags