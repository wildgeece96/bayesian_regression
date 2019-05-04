import numpy as np

# 基底関数
def base_polynomial(x, M=5):
    """
    多項式基底を元にした計画行列の作成
    inputs:
        x : 2d-array. Nxd  = (サンプル数)x(特徴量数)
    return:
        Phi : 2d-array. Nx(Mxd). (1つの特徴量に対してM個の基底関数を適用)
                計画行列を返す
    """
    N = x.shape[0] # サンプル数
    d = x.shape[1] # 特徴量数
    Phi = np.zeros(shape=(N, int(M*d)), dtype='float')
    for m in range(M):
        Phi[:,m::M] = x**m
    return Phi

def base_gauss(x, M=5, seed=42, mu=None):
    """
    ガウス基底を元にした計画行列の作成
    inputs:
        x : 2d-array. Nxd  = (サンプル数)x(特徴量数)
    return:
        Phi : 2d-array. Nx(Mxd). (1つの特徴量に対してM個の基底関数を適用)
                計画行列を返す
    今回は簡略化のため、σの値は1で統一する(これは入力が正規化されてるのも理由の1つ)
    """
    np.random.seed(seed)
    N = x.shape[0]
    d = x.shape[1]
    if type(mu) != np.ndarray:
        mu = np.random.randn(M*d) # 基底関数の数だけ平均値をランダム生成
    else:
        M = mu.shape[0]//d
    x_tile = np.tile(x, (1,M))
    Phi = np.exp(-(x_tile-mu)**2/2)
    return Phi, mu

def bayesian_regression(X, y, base_mode='polynomial',M=5, w_S=None, w_m=None, mu=None,beta=1e0):
    """
    ベイズ推定を行う。
    inputs:
        X : 2d-array(Nxd). 説明変数
        y : 1d-array(N,). 教師データ
        base_mode: {'polynomial', 'gauss'}. 基底関数の種類
        M : 各説明変数に対して何個ずつ基底関数を生成するか.
        mu : ガウス分布を基底関数として使うときの各々の分布の平均値
        beta: 精度(分散の逆数)
    return:
        wm_list : 2d-array.((Mxd), 1)のlist. 重みベクトルの期待値のリスト
        wS_list : 2d-array.((Mxd), (Mxd))のlist. 重みベクトルの分散のリスト
        m_list : 2d-array. (N, 1)のlist. 予測値の期待値のリスト
        S_list : 2d-array. (N,N)のlist. 予測値の分散のリスト
    """
    d = X.shape[1]
    N = X.shape[0]
    if not w_S:
        w_S = np.eye(M*d)
    if not w_m:
        w_m = np.zeros(M*d).reshape(-1,1)

    wS_list = []
    wm_list = []
    S_list = []
    m_list = []
    lam = 1e-3 # 逆行列を求める時にエラーを防ぐ
    if base_mode == 'polynomial':
        Phi_all = base_polynomial(X, M)
    elif base_mode == 'gauss':
        Phi_all, mu = base_gauss(X, M, mu=mu)
    else:
        raise ValueError("Please valid base mode , {'gauss', 'polynomial'}")
    for i in range(N):
        w_S_0 = w_S +  lam*np.eye(M*d)
        w_m_0 = w_m +  lam*np.ones(M*d).reshape(-1,1)
        if base_mode=='polynomial':
            Phi = base_polynomial(X[i,:].reshape(1,-1),M)
        elif base_mode == 'gauss':
            Phi, mu = base_gauss(X[i,:].reshape(1,-1), M=M, mu=mu)
        w_S = np.linalg.inv(np.linalg.inv(w_S_0) + beta * np.dot(Phi.T,Phi))
        m_partial_1 = np.dot(np.linalg.inv(w_S_0), w_m_0)
        m_partial_2 = beta * np.dot(Phi.T, y[i].reshape(1,1))
        w_m = w_S.dot(m_partial_1+m_partial_2)
        m = np.dot(Phi_all, w_m)
        S = beta**(-1) + np.dot(Phi_all, np.dot(w_S, Phi_all.T))
        wS_list.append(w_S)
        wm_list.append(w_m)
        S_list.append(S)
        m_list.append(m)
    return wS_list,wm_list,S_list,m_list
