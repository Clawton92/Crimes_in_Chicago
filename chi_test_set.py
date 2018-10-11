import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics import regressionplots as smg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression


def pd_concat_sampled_df(df, col, val1, val2, sample_size):
    query_string1 = "{} == {}".format (col, val1)
    query_string2 = "{} == {}".format(col, val2)
    first = df.query(query_string1)
    second = df.query(query_string2)
    samp1 = first.sample(sample_size)
    samp2 = second.sample(sample_size)
    samp1.index = range(sample_size)
    samp2.index = range(sample_size)
    frames = [samp1, samp2]
    final = pd.concat(frames)
    final_shuffle = final.sample(frac=1).reset_index(drop=True)
    return final_shuffle

def logit_kfold(X, y, folds):
    kf = KFold(folds, shuffle=True)
    accuracies = []
    precisions = []
    recalls = []

    for train_index, test_index in kf.split(X):
        model = LogisticRegressionCV(cv=10)
        model.fit(X[train_index], y[train_index])
        y_predict = model.predict(X[test_index])
        y_true = y[test_index]
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))
    kf_scores = ("accuracy: {}".format(np.average(accuracies)), "precision: {}".format(np.average(precisions)),\
        "recall: {}".format(np.average(recalls)))
    return kf_scores

def roc_c(x_vals, y_vals, feature_title, save_title):
    X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals, stratify=y_vals)
    model = LogisticRegressionCV(cv=10)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = round(auc(fpr, tpr), 2)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.text(0.8, 0.0, 'auc = {}'.format(roc_auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC plot of arrests in Chicago: {}".format(feature_title))
    # plt.savefig('graphs/{}'.format(save_title))
    plt.show()

def lasso_cv(x_vals, y_vals):
    X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals, stratify=y_vals)
    model = LogisticRegression(penalty='l1').fit(X_train, y_train)
    return model

def lasso_cross_val(x_vals, y_vals):
    clf = LogisticRegressionCV(cv=10, random_state=0, multi_class='multinomial').fit(x_vals, y_vals)
    return clf

def remove_cols(df_cols, coeffs):
    drop_vals = [element for idx, element in enumerate(df_cols) if coeffs[0][idx] == 0]
    return drop_vals

def keep_coeffs(df_cols, coeffs):
    keep_vals = [coeffs[0][idx] for idx, element in enumerate(df_cols) if coeffs[0][idx] != 0]
    return keep_vals


def vif(X):
    for idx, col in enumerate(X.columns):
        print (f"{col}: {oi.variance_inflation_factor(X.values,idx)}")


if __name__=='__main__':
    #reading in new data sets after mistake
    five_feat_balanced = pd.read_csv('data_sets/five_feat_balanced.csv')
    five_feat_balanced.drop('Unnamed: 0', axis=1, inplace=True)
    five_feat_balanced.sort_values('arrest', inplace=True)
    five_short_head = five_feat_balanced.iloc[0:1000]
    five_short_tail = five_feat_balanced.iloc[five_feat_balanced.shape[0]-1000:five_feat_balanced.shape[0]]
    five_feat_short = pd.concat([five_short_head, five_short_tail])
    five_feat_short = five_feat_short.sample(frac=1).reset_index(drop=True)

    full_feat_balanced = pd.read_csv('data_sets/full_feat_balanced.csv')
    full_feat_balanced.drop('Unnamed: 0', axis=1)
    full_feat_balanced.sort_values('arrest', inplace=True)
    full_short_head = full_feat_balanced.iloc[0:1000]
    full_short_tail = full_feat_balanced.iloc[full_feat_balanced.shape[0]-1000:full_feat_balanced.shape[0]]
    full_feat_short = pd.concat([full_short_head, full_short_tail])
    full_feat_short = full_feat_short.sample(frac=1).reset_index(drop=True)
    full_feat_short.drop('Unnamed: 0', axis=1, inplace=True)
    full_feat_long = full_feat_balanced[1000:full_feat_balanced.shape[0]-1000]

    '''new data starts here'''
    #roc for 5 feature test set
    roc_c(five_feat_short.drop('arrest', axis=1).values, five_feat_short['arrest'].values, 'retest 5 feat', 'testing')

    #roc for full feature test set
    roc_c(full_feat_short.drop('arrest', axis=1).values, full_feat_short['arrest'].values, 'retest full feat', 'testing full')
    ##lasso on test data. Saving coeffs to csv for presentation
    # lasso_log = lasso_cv(full_feat_short.drop('arrest', axis=1).values, full_feat_short['arrest'].values)
    # ls = [lasso_log.coef_[0][idx] for idx in range(lasso_log.coef_.shape[1])]
    # df_coeffs = pd.DataFrame({'coeffs': ls})
    # df_coeffs.to_csv('data_sets/df_coeffs.csv')
    #
    # #removing columns for new data sets with fixed features
    # columns_to_remove = remove_cols(full_feat_short.drop('arrest', axis=1).columns, lasso_log.coef_)
    # full_feat_short_final = full_feat_short.drop(columns_to_remove, axis=1)
    # full_feat_short_final.to_csv('data_sets/full_feat_short_final.csv')
    #
    # full_feat_long_final = full_feat_long.drop(columns_to_remove, axis=1)
    # full_feat_long_final.to_csv('data_sets/full_feat_long_final.csv')

    #reading in final csvs
    full_short_final = pd.read_csv('data_sets/full_feat_short_final.csv')
    full_short_final.drop('Unnamed: 0', axis=1, inplace=True)
    #vif for new dataframe after lasso elimination
    vif(full_short_final.drop('arrest', axis=1))
    roc_c(full_short_final.drop('arrest', axis=1).values, full_short_final['arrest'].values, 'full short final!', 'testing full')
    logit_kfold(X, y, folds)

    full_long_final = pd.read_csv('data_sets/full_feat_long_final.csv')
    full_long_final.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    roc_c(full_long_final.drop('arrest', axis=1).values, full_long_final['arrest'].values, 'full long final!', 'testing full')
    logit_kfold(X, y, folds)
