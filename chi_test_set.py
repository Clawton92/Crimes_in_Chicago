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

#test data frame with 5 features and 1000 rows (500 arrested, 500 non arrests)
test_set = pd.read_csv('data_sets/crimes-in-chicago/test_set.csv')
df_shuffle = test_set.sample(frac=1).reset_index(drop=True)
y_initial = df_shuffle['arrest'].values
x_initial = df_shuffle[['domestic', 'loc_north', 'loc_south', 'crime_non_violent', 'crime_violent']].values

#2nd test data set with full dummy variables for wards and crimes types, this also included domestic
#2000 rows, 1000 arrested and 1000 non arrested. This dataframe is already shuffled
df_dummies_test = pd.read_csv('data_sets/crimes-in-chicago/full_dummies_set.csv')
df_dummies_test.drop('Unnamed: 0', axis=1, inplace=True)

y_dummies = df_dummies_test['arrest'].values
X_dummies = df_dummies_test.drop('arrest', axis=1).values
df_X_dummies = df_dummies_test.drop('arrest', axis=1)

#reading in final set to train
df_final_full = pd.read_csv('data_sets/crimes-in-chicago/final_set_to_train.csv')

def pd_concat_sampled_df(df, col, val1, val2, sample_size):
    query_string1 = "{} == {}".format(col, val1)
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
        model = LogisticRegression()
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
    X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals)
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
    X_train, X_test, y_train, y_test = train_test_split(x_vals, y_vals)
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
    #kfold for initial 5 feature model
    kf_5feat = logit_kfold(x_initial, y_initial, 10)

    #roc curve for initial 5 feature dataframe
    # roc_c(x_initial, y_initial, 'test set with 5 features', 'roc_test_5_features')

    #creating lasso with l1 penalty and finding zero coeffs to drop and writing to csv
    # lasso_log = lasso_cv(X_dummies, y_dummies)
    # #writing coeffs to dataframe
    # ls = [lasso_log.coef_[0][idx] for idx in range(lasso_log.coef_.shape[1])]
    # df_coeffs = pd.DataFrame({'coeffs': ls})
    # df_coeffs.to_csv('data_sets/df_coeffs.csv')
    #
    # #X_val Dataframe for dummies test model
    # columns_to_remove = remove_cols(df_X_dummies.columns, lasso_log.coef_)
    # df_reduced_dummies = df_X_dummies.drop(columns_to_remove, axis=1)
    # df_reduced_dummies.to_csv('data_sets/df_reduced_dummies.csv')

    #reading in reduced dummiers df
    df_reduced_dummies = pd.read_csv('data_sets/df_reduced_dummies.csv')
    # # vif(df_reduced_dummies)
    #
    # # #kfold for initial 28 feature model
    # # kf_28feat_initial = logit_kfold(df_X_dummies.values, y_dummies, 10)

    #roc curve for initial reduced feature dataframe
    roc_c(df_reduced_dummies.values, y_dummies, 'test set with new features', 'test_set_28_features')

    # #creating undersampled and reduced full dataframe for years 2012-2017
    # df_final_balanced_classes = pd_concat_sampled_df(df_final_full, 'arrest', 1, 0, df_final_full.query('arrest == 1').shape[0])
    # df_final_balanced_classes.drop('Unnamed: 0', axis=1, inplace=True)
    # final_cols_to_drop = remove_cols(df_final_balanced_classes.drop('arrest', axis=1).columns, lasso_log.coef_)
    # df_final_balanced_reduced = df_final_balanced_classes.drop(final_cols_to_drop, axis=1)
    # df_final_balanced_reduced.to_csv('data_sets/df_final_balance_reduced.csv')

    # read in final reduced df
    df_final_balanced_reduced = pd.read_csv('data_sets/df_final_balance_reduced.csv')
    # #kfold for final 28 feature model
    # kf_28feat_final = logit_kfold(df_final_balanced_reduced.drop('arrest', axis=1).values, df_final_balanced_reduced['arrest'].values, 10)

    #roc curve for final set
    roc_c(df_final_balanced_reduced.drop('arrest', axis=1).values, df_final_balanced_reduced['arrest'].values, 'full set with new features', 'roc_full_set_28_features')
    #
