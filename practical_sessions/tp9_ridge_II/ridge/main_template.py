"""
    Statistical comparison between Ridge regression estimator and OLS.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from utils_algo import compute_lambda_star_and_risk_star, ridge_risk
from constants import BAYES_RISK, SIGMA, SEED, N_TESTS

def main():
    n = 30
    d_list = [10, 20, 30]
    llambda_list = [10**(n) for n in np.arange(-8, 5, 0.2)]

    risks = dict()
    llambda_stars_risks = dict()
    infinity_biases = dict()

    # loop over parameters
    for llambda in llambda_list:
        for d in d_list:
            # Load design matrix
            X_path = os.path.join("data", f"n_design_matrix_n={n}_d={d}.npy")
            X = np.load(X_path)
            n = X.shape[0]

            """
            add operations here
            """

            print(f"\nlambda: {llambda}")
            print(f"d: {d}")
            #  risk of the ridge estimator
            risks[(llambda, d)] = 1
            # lambda_star and the corresponding risk
            llambda_stars_risks[d] = (1, 1)
            # compute bias limit when llambda is large
            infinity_biases[d] = 1


    colors = ["blue", "green", "darkred", "mediumvioletred", "darkmagenta"]
    index = 0
    for d in d_list:
        color = colors[index]
        risk_estimates = [risks[llambda, d] for llambda in llambda_list]
        # plot lambda_star and the corresponding risk
        llambda_star, risk_star = llambda_stars_risks[d]
        plt.plot(llambda_star, risk_star, "x", color=color, markersize=12, label = r"$\lambda^*$"+f", d={d}")
        infinity_bias = infinity_biases[d]
        alpha = 0.4
        if index == 0:
            label_est = f"risk estimation, d={d}"
            plt.plot(llambda_list,
                     risk_estimates,
                     "o",
                     label=label_est,
                     color=color,
                     markersize=3,
                     alpha=alpha)
            plt.plot(llambda_list,
                     [BAYES_RISK+SIGMA**2*d/n]*len(llambda_list),
                     label="OLS risk: "+r"$\sigma^2+\frac{\sigma^2d}{n}$"+f", d={d}",
                     color=color,
                     alpha = alpha)
            plt.plot(llambda_list,
                     [SIGMA**2+infinity_bias]*len(llambda_list),
                     label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}",
                     color=color,
                     alpha = 0.8*alpha,
                     linestyle="dashed")
        else:
            label_est = f"d={d}"
            plt.plot(llambda_list, risk_estimates, "o", label=label_est, color=color, markersize=3, alpha=alpha)
            plt.plot(llambda_list, [BAYES_RISK+SIGMA**2*d/n]*len(llambda_list), color=color, alpha=alpha)
            plt.plot(llambda_list, [SIGMA**2+infinity_bias]*len(llambda_list),
                     label=r"$Risk_{\lambda\rightarrow +\infty}$"+f", d={d}",
                     color=color, alpha = 0.8*alpha, linestyle="dashed")
        index += 1

    # finalize plot
    plt.xlabel(r"$\lambda$")
    plt.xscale("log")
    plt.ylabel("risk")
    plt.plot(llambda_list, [BAYES_RISK]*len(llambda_list), label="Bayes risk: "+r"$\sigma^2$", color="aqua")
    title = (
        "Ridge regression: test error as a function of "
        r"$\lambda$"
        f" and d\nn={n}")
    plt.title(title)
    plt.legend(loc="best", prop={"size": 6})
    plt.tight_layout()
    figname = f"test_errors_n={n}_r_state_{SEED}.pdf"
    figpath = os.path.join("images", figname)
    plt.savefig(figpath)


if __name__ == "__main__":
    main()
