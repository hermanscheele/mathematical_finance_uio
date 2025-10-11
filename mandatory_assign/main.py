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
        r_lst = get_returns_vec(assets, df)
        cov_matrix = cov_mat(r_lst)
        mu = np.array(est_return_vec(assets, df))

        plot_efficient_frontier(mu, cov_matrix, r_lst)
        w_m, mu_m, sigma_m = min_var_portfolio(mu, cov_matrix)
        print(f'Min. Var. Portfolio: w = {w_m}, return = {mu_m}, risk = {sigma_m}')

        plt.legend()
        plt.show()


    def p1d(self):
    
        for i in range(2):
            r_lst = get_returns_vec(assets, df)[i:]
            mu = np.array(est_return_vec(assets, df)[i:])

            cov_matrix = cov_mat(r_lst)

            plot_efficient_frontier(mu, cov_matrix, r_lst)
    
        plt.legend()
        plt.show()

    
    def p2a(self):
        return 0    






a = Assignment()

# a.p1a()
# a.p1b()
# a.p1c()
a.p1d()