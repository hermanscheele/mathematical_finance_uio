import pandas as pd
import math as m
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import brentq



def get_returns(price_data):
    r = []
    n = len(price_data)

    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))
        r.append( (s_i_1 - s_i) / s_i )

    return r


def get_returns_vec(assets, df):
    l = []
    for a in assets:
        p = df[a]
        l.append(get_returns(p))
    return l


def est_return(r):
    return sum(r) / len(r)


def est_return_vec(assets, df):
    vec = []
    for a in assets:
        p = df[a]
        r = get_returns(p)
        vec.append(est_return(r))
    return vec


def est_volatiliy(r):
    n = len(r)
    s = 0.0
    for i in range(n):
        s += (r[i] - est_return(r)) ** 2
    
    return np.sqrt((1 / (n - 1)) * s)


def fit_norm_dist(x, mu, sigma):
    return (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-(1/2) * ((x - mu)**2 / sigma**2))


def plot_return_dist(r, mu, sigma, a):

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = fit_norm_dist(x, mu, sigma)

    plt.plot(x, y, color='r', label='fitted normal dist.')
    sns.histplot(r, color="skyblue", label='empirical dist.')

    plt.title(f"Asset: {a}")
    plt.legend()
    plt.show()



# Covariance
def cov(r_i, r_j):

    n = len(r_i)
    s = 0.0
    for k in range(n):
        s += ( (r_i[k] - est_return(r_i)) * (r_j[k] - est_return(r_j)) )

    return (1 / (n - 1)) * s

# Correlation 
def corr(r_i, r_j):
    return cov(r_i, r_j) / (est_volatiliy(r_i) * est_volatiliy(r_j))


# Covariance matrix
def cov_mat(r):
    n = len(r)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cov(r[i], r[j]))
        matrix.append(row)

    return np.array(matrix)


# Markowitz scalars
def markowitz_scalars(mu, cov):
    n = len(cov)
    C_inv = np.linalg.inv(cov)
    e = np.ones(n)

    A = e.T @ C_inv @ e
    B = e.T @ C_inv @ mu
    C = mu.T @ C_inv @ mu
    D = A * C - (B**2)

    return A, B, C, D


# Calculate the min. var. and plot the efficient frontier
def plot_efficient_frontier(mu, cov, r_lst):

    A, B, C, D = markowitz_scalars(mu, cov)

    def min_var(x):
        return ((A * x**2) - (2*B * x) + C) / D

    y = np.linspace(-min(mu), max(mu), 50)
    x = np.sqrt(min_var(y))

    plt.plot(x, y, label=f'Efficient Frontier n = {len(r_lst)}')
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    
    for r in r_lst:
        plt.plot(est_volatiliy(r), est_return(r), 'o')


# w, mu & sigma for the min. var. portfolio
def min_var_portfolio(mu, cov):

    A, B, _, _ = markowitz_scalars(mu, cov)
    n = len(cov)
    C_inv = np.linalg.inv(cov)
    e = np.ones(n)

    # Min. Var. Portfolio
    w_m = (1/A) * C_inv @ e
    mu_m = B / A
    sigma_m = 1 / np.sqrt(A)
    
    return w_m, mu_m, sigma_m



def black_scholes_call(s_0, k, r, t, sigma):

    def N(d):
        return 0.5 * (1 + m.erf(d / m.sqrt(2)))
    
    d_1 = (m.log(s_0 / k) + (r + (0.5 * sigma**2)) * t) / (sigma * m.sqrt(t))
    d_2 = d_1 - (sigma * m.sqrt(t))

    return (s_0 * N(d_1)) - (k * m.exp(-r * t) * N(d_2))



def implied_vol_call(C_mkt, s0, k, r, t, lo=1e-9, hi=5.0):
    f = lambda sig: black_scholes_call(s0, k, r, t, sig) - C_mkt
    return brentq(f, lo, hi)  