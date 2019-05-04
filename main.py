# -*- coding: utf-8 -*-
import numpy as np
from utils import *
from argparse import ArgumentParser
import os
mode = 'polynomial'
M = 5

argparser = ArgumentParser()
argparser.add_argument('--M', type=int,\
            help='The number of base function per feature')
argparser.add_argument('--mode', type=str,\
            help='The kind of base function {"gauss", "polynommial"}')
args = argparser.parse_args()

if args.M:
    M = args.M
if args.mode in ['polynomial', 'gauss']:
    mode = args.mode
# 訓練データの作成
N = 30
d = 1
X = np.linspace(-4, 4, N).reshape(-1,1)
y_true = 0.3*np.sin(X)+ 0.4*np.cos(X)
y = y_true +np.random.randn(N,1)/3.0
X = (X - X.mean(axis=0))/X.std(axis=0)
y = (y - y.mean())/y.std()
y_true = (y_true - y_true.mean())/y_true.std()

if not "fig" in os.listdir("./"):
    os.mkdir("fig")
from matplotlib import pyplot as plt
# base_modeの指定
if mode == 'gauss':
    mu = np.tile(np.linspace(X.min(), X.max(),M),d)
wS_list,wm_list, S_list, m_list = bayesian_regression(X, y,M=M,beta=1.0,
                                                      base_mode=mode)
for i in range(X.shape[0]):
    plt.figure(figsize=(12,8))
    plt.ylim(-2.5,2.5)
    plt.xlim(-2.5,2.5)
    SD = np.sqrt(np.diag(S_list[i]))
    upper = m_list[i].flatten() + SD.flatten()
    lower = m_list[i].flatten() - SD.flatten()
    # 分布のプロット
    plt.fill_between(X.flatten(), upper.flatten(), lower.flatten(), color='pink')
    # 今回使ったデータのプロット
    plt.scatter(X.flatten(),y.flatten(), label='raw_data')
    # 予測値の平均(期待値)のプロット
    plt.plot(X.flatten(), m_list[i].flatten(), label='pred_mean',color='b')
    # 元々のモデルのプロット
    plt.plot(X.flatten(), y_true, label='ideal_model', color='r')
    plt.legend()
    plt.title(f"{mode}:{i}")
    plt.savefig(f'fig/bayesian_{mode}_{i}.png')
