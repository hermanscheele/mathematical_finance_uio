from utils import *
import pandas as pd


df = pd.read_csv("finance_data.csv", sep=";")
assets = df.columns.tolist()[1:]
n = len(assets)


class Assignment:

    def p1a(self):
        for a in assets:
            p = df[a] # price data
            r = get_returns(p) # returns for asset a

            r_hat = est_return(r)
            sigma_hat = est_volatiliy(r)

            print(f'Asset: {a}, r_hat: {r_hat}, sigma_hat: {sigma_hat}')    
            plot_return_dist(r, r_hat, sigma_hat)


    def p1b(self):
        corrs = []
        r_lst = []

        for i in range(n):
            r_i = get_returns(df[assets[i]])
            r_lst.append(r_i)
            for j in range(n):
                r_j = get_returns(df[assets[j]])
                c = corr(r_i, r_j)

                print(f'Corr(R{i}, R{j}) = {c}')
                
        covar_matrix = cov_mat(r_lst)
        print(covar_matrix)


    def p1c(self):
        return 0


a = Assignment()

# a.p1a()
# a.p1b()