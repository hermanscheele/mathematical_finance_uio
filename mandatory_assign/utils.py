import pandas as pd
import math as m
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_returns(price_data):
    r = []
    n = len(price_data)

    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))
        r.append( (s_i_1 - s_i) / s_i )

    return r


def est_return(r):
    return sum(r) / len(r)


def est_volatiliy(r):
    n = len(r)
    s = 0.0
    for i in range(n):
        s += (r[i] - est_return(r)) ** 2
    
    return np.sqrt((1 / (n - 1)) * s)



# #--------------- need this for the fitted normal distribution -------------------- #
# def all_est_return_vol(data, df):
#     est_return_vol_list = []
#     for i in range(1, len(data)):
#         prices = df[data[i]]
#         t = (est_return(prices), est_volatiliy(prices))
#         est_return_vol_list.append(t)

#     return est_return_vol_list


def fit_norm_dist(x, mu, sigma):
    return (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-(1/2) * ((x - mu)**2 / sigma**2))


def plot_return_dist(r, mu, sigma):

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = fit_norm_dist(x, mu, sigma)

    plt.plot(x, y, color='r', label='fitted normal dist.')
    sns.histplot(r, color="skyblue", label='empirical dist.')

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


