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


def remove_ret(tpl, elem):
    """this helps to remove one element from a tuple"""
    lst = list(tpl)
    lst.remove(elem)
    return tuple(lst)

def group_linear_and_squared(feature_names):
    """
    Groups features so that e.g. 'x1' and 'x1^2' are treated as one predictor.
    Returns: list of tuples, where each tuple contains indices for a predictor (linear + optional squared).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    used = set()
    groups = []

    for i, name in enumerate(feature_names):
        if name in used:
            continue
        # Handle only the base (linear) term or stand-alone squared terms
        if not name.endswith("^2"):
            sq_name = f"{name}^2"
            if sq_name in name_to_idx:
                sq_idx = name_to_idx[sq_name]
                groups.append((i, sq_idx))
                used.update([i, sq_idx])
            else:
                groups.append((i,))
                used.add(i)
        else:
            # Handle orphan squared terms (no matching linear)
            base = name[:-2]
            if base not in name_to_idx:
                groups.append((i,))
                used.add(i)

    return groups


def dominance_stats(X, y, feature_names=None):
    """
    Return the dominance analysis statistics for multilinear regression.

    see also: https://github.com/dominance-analysis/dominance-analysis and
    https://netneurotools.readthedocs.io/en/stable/generated/netneurotools.stats.get_dominance_stats.html

    if the regression includes quadratic terms the x^2 is treated as part of x 

    Parameters
    ----------
     -X: ((N, M) array_like) – Input data
     -y ((N,) array_like) – Target values
     -feature_names (list of str, optional) – Names of features. Used to detect quadratic terms automatically.

    Returns
    -------
    -model_metrics (dict) – The dominance metrics, containing individual_dominance, partial_dominance, total_dominance, and full_r_sq
    """
    if feature_names is not None:
        predictor_groups = group_linear_and_squared(feature_names)
    else:
        predictor_groups = [(i,) for i in range(X.shape[1])]

    n_lin = len(predictor_groups)

    # generate all predictor combinations in list (using linear terms only)
    predictor_combs = [list(combinations(range(n_lin), i))
                       for i in range(1, n_lin + 1)]

    # get all r2
    model_r_sq = dict()
    for len_group in predictor_combs:
        for idx_tuple in len_group:
            # get full index set for selected predictors (include squared terms)
            full_indices = []
            for pred_idx in idx_tuple:
                full_indices.extend(predictor_groups[pred_idx])
            full_indices = tuple(sorted(full_indices))
            lin_reg = LinearRegression()
            X_indx = X[:, full_indices]
            lin_reg.fit(X_indx, y)
            yhat = lin_reg.predict(X_indx) 
            SS_Residual = sum((y - yhat) ** 2)
            SS_Total = sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            model_r_sq[full_indices] = r_squared 

    model_metrics = dict([])

    # individual dominance for each predictor 
    individual_dominance = []
    for i_pred in range(n_lin):
        key = tuple(sorted(predictor_groups[i_pred]))
        individual_dominance.append(model_r_sq[key])
    individual_dominance = np.array(individual_dominance).reshape(1, -1)
    model_metrics["individual_dominance"] = individual_dominance

    # partial dominance for each predictor 
    partial_dominance = [[] for _ in range(n_lin - 1)] #combiantions start at 2
    for i_len in range(n_lin - 1): #loop through possible n of predictor
        i_len_combs = list(combinations(range(n_lin), i_len + 2)) #combinations of predictors possible with this n 
                                                                  #(only sesible with at least 2 predictors, otherwise it's just individual dominance)
        for j_node in range(n_lin): #loop through predictors 
            j_node_sel = [v for v in i_len_combs if j_node in v] #list containing only combinations that include current predictor
            reduced_list = [remove_ret(comb, j_node) for comb in j_node_sel] #same combinations without the current predictor
            diff_values = []
            for i in range(len(reduced_list)):
                full_idx = []
                red_idx = []
                for pred in j_node_sel[i]:
                    full_idx.extend(predictor_groups[pred])
                for pred in reduced_list[i]:
                    red_idx.extend(predictor_groups[pred])
                full_idx = tuple(sorted(full_idx))
                red_idx = tuple(sorted(red_idx))
                diff_values.append(model_r_sq[full_idx] - model_r_sq[red_idx]) #get difference in r2 for each combination
            partial_dominance[i_len].append(np.mean(diff_values)) #mean effect of including each predictor at this cobination length
  
    # save partial dominance
    partial_dominance = np.array(partial_dominance)
    model_metrics["partial_dominance"] = partial_dominance
    # get total dominance
    total_dominance = np.mean(
        np.r_[individual_dominance, partial_dominance], axis=0) #mean across all possible lengths (including as single predictor)
    model_metrics["total_dominance"] = total_dominance
    # save r2
    all_indices = []
    for group in predictor_groups:
        all_indices.extend(group)
    all_indices = tuple(sorted(all_indices))
    model_metrics["full_r_sq"] = model_r_sq[all_indices] #retrieve full model

    return model_metrics