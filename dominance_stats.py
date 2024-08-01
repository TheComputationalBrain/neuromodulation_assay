import os
#set threads to 1 before importing numpy to prevent unwanted behavior when running on the server
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression


# this helps to remove one element from a tuple
def remove_ret(tpl, elem):
    lst = list(tpl)
    lst.remove(elem)
    return tuple(lst)

def dominance_stats(X, y):
    """
    Return the dominance analysis statistics for multilinear regression.

    see also: https://github.com/dominance-analysis/dominance-analysis

    Parameters
    ----------
     -X: ((N, M) array_like) – Input data
     -y ((N,) array_like) – Target values

    Returns
    -------
    -model_metrics (dict) – The dominance metrics, containing individual_dominance, partial_dominance, total_dominance, and full_r_sq
    """
    # generate all predictor combinations in list 
    n_predictor = X.shape[-1]
    # n_comb_len_group = n_predictor - 1
    predictor_combs = [list(combinations(range(n_predictor), i))
                       for i in range(1, n_predictor + 1)]

    # get all r2
    model_r_sq = dict()
    for len_group in predictor_combs:
        for idx_tuple in len_group:
            lin_reg = LinearRegression()
            X_indx = X[:, idx_tuple]
            lin_reg.fit(X_indx, y)
            yhat = lin_reg.predict(X_indx) 
            SS_Residual = sum((y - yhat) ** 2)
            SS_Total = sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            model_r_sq[idx_tuple] = r_squared 

    model_metrics = dict([])

    # individual dominance for each predictor 
    individual_dominance = []
    for i_pred in range(n_predictor):
        individual_dominance.append(model_r_sq[(i_pred,)])
    individual_dominance = np.array(individual_dominance).reshape(1, -1)
    model_metrics["individual_dominance"] = individual_dominance

    # partial dominance for each predictor 
    partial_dominance = [[] for _ in range(n_predictor - 1)] #combiantions start at 2
    for i_len in range(n_predictor - 1): #loop through possible n of predictor
        i_len_combs = list(combinations(range(n_predictor), i_len + 2)) #combinations of predictors possible with this n 
                                                                        #(only sesible with at least 2 predictors, otherwise it's just individual dominance)
        for j_node in range(n_predictor): #loop through predictors 
            j_node_sel = [v for v in i_len_combs if j_node in v] #list containing only combinations that include current predictor
            reduced_list = [remove_ret(comb, j_node) for comb in j_node_sel] #same combinations without the current predictor
            diff_values = [
                model_r_sq[j_node_sel[i]] - model_r_sq[reduced_list[i]] #get difference in r2 for each combination
                for i in range(len(reduced_list))]
            partial_dominance[i_len].append(np.mean(diff_values)) #mean effect of including each predictor at this cobination length
  
    # save partial dominance
    partial_dominance = np.array(partial_dominance)
    model_metrics["partial_dominance"] = partial_dominance
    # get total dominance
    total_dominance = np.mean(
        np.r_[individual_dominance, partial_dominance], axis=0) #mean across all possible lengths (including as single predictor)
    model_metrics["total_dominance"] = total_dominance
    # save r2
    model_metrics["full_r_sq"] = model_r_sq[tuple(range(n_predictor))] #retrieve full model

    return model_metrics