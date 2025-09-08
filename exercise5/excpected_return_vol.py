import pandas as pd
import math as m
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("finance_data.csv", sep=";")

cols = df.columns.tolist()


def est_return(price_data):

    n = len(price_data) 
    r_sum = 0.0
    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))

        r_i = (s_i_1 - s_i) / s_i
        r_sum += r_i

    r_k = r_sum / n
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

    sigma = m.sqrt(s / (n - 1))
    return sigma



#--------------- need this for the fitted normal distribution -------------------- #
def all_est_return_vol(data):
    est_return_vol_list = []
    for i in range(1, len(data)):
        prices = df[data[i]]
        t = (est_return(prices), est_volatiliy(prices))
        est_return_vol_list.append(t)

    return est_return_vol_list




def plot_return_data(price_data, r_k, sigma):

    n = len(price_data)
    r_is = []
    for i in range(n - 1):
        s_i = float(price_data[i].replace(',', "."))
        s_i_1 = float(price_data[i + 1].replace(',', "."))

        r_i = (s_i_1 - s_i) / s_i
        r_is.append(r_i)


    #----------- make and plot the fitted normal distribution --------------------#

    sns.histplot(r_is, color="skyblue")
    sns.kdeplot(r_is, color="red")

    plt.legend()
    plt.show()
    

energy_pricedata = df[cols[1]]
plot_return_data(energy_pricedata, est_return(energy_pricedata), est_volatiliy(energy_pricedata))
