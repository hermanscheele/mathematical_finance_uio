from utils import *
import pandas as pd


df = pd.read_csv("finance_data.csv", sep=";")
assets = df.columns.tolist()[1:]


# ------ Prob. 1 a) ------ #
for a in assets:
    p = df[a] # price data
    r_hat = est_return(p)
    sigma_hat = est_volatiliy(p)

    print(f'Asset: {a}, r_hat: {r_hat}, sigma_hat: {sigma_hat}')    
    plot_return_dist(p) # make normal fitted dist <<-----------------------------------


  
