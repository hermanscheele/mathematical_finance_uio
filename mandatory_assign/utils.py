import pandas as pd
import math as m
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def est_return(price_data):

    n = len(price_data) 
    r_sum = 0.0
    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))

        r_i = (s_i_1 - s_i) / s_i
        r_sum += r_i

    r_k = r_sum / (n - 1) # n price points p, (n - 1) returns r
    return r_k


def est_volatiliy(price_data):

    r_k = est_return(price_data)
    n = len(price_data)
    s = 0.0
    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))

        r_i = (s_i_1 - s_i) / s_i
        s += (r_i - r_k) ** 2

    sigma = m.sqrt(s / (n - 2))
    return sigma



#--------------- need this for the fitted normal distribution -------------------- #
def all_est_return_vol(data, df):
    est_return_vol_list = []
    for i in range(1, len(data)):
        prices = df[data[i]]
        t = (est_return(prices), est_volatiliy(prices))
        est_return_vol_list.append(t)

    return est_return_vol_list




#----------- make and plot the fitted normal distribution --------------------#
def plot_return_dist(price_data, r_k, sigma):

    n = len(price_data)
    r_is = []
    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))

        r_i = (s_i_1 - s_i) / s_i
        r_is.append(r_i)

    sns.histplot(r_is, color="skyblue")
    

    plt.legend()
    plt.show()
