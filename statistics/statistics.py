#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File : statistics.py
# @Author : Martin Yan
# @Time : 2021/3/21 下午1:53
# @Software : PyCharm

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def chi_squared_test(data, target):
    """
    Compute the chi2 and p values of categorical attributes to the target and output the attributes
    with p less than 0.05
    :param data: categorical attributes with the target
    :param target: the target attribute
    :return: dataframe of attributes with p less than 0.05
    """
    variable = []
    score = []
    p = []
    x = data.loc[:, data.columns != target]
    y = data.loc[:, data.columns == target]
    for attr in x:
        model = SelectKBest(chi2, k=1)
        model.fit_transform(x[[attr]], y)
        if model.pvalues_[0] < 0.05:
            variable.append(attr)
            score.append(model.scores_)
            p.append(model.pvalues_)

    result = pd.DataFrame({'Variable': variable, 'Score': score, 'P': p})
    return result


def t_test_with_levene(data, target):
    """
    Perform Levene test for equal variances and next calculate the T-test for the means of two independent
    samples of scores and finally output the attributes with p less than 0.05
    :param data: numerical attributes with the target
    :param target: the target attribute
    :return: dataframe of attributes with p less than 0.05
    """
    variable = []
    score = []
    p = []
    for attr in data:
        if attr != target:
            data1 = data[data[target] == 0][attr]
            data2 = data[data[target] == 1][attr]
            equal_var = True
            if stats.levene(data2, data1, center='median')[1] < 0.05:
                equal_var = False
            # If True (default), perform a standard independent 2 sample test (student t) that assumes equal population
            # variances. If False, perform Welch’s t-test, which does not assume equal population variance
            t_and_p = stats.ttest_ind(data2, data1, equal_var=equal_var)
            if t_and_p[1] < 0.05:
                variable.append(attr)
                score.append(t_and_p[0])
                p.append(t_and_p[1])

    result = pd.DataFrame({'Variable': variable, 'Score': score, 'P': p})
    return result


def check_normality(data, attrs, type='pp'):
    """
    Check the distribute whether is normal distribution
    :param data: origin data
    :param attrs: attributes to be tested
    :param type: pp or qq
    :return: results of attributes
    """
    variable = []
    normal = []
    p = []
    for attr in attrs:
        variable.append(attr)
        print(stats.kstest(data[attr], 'norm')[1])
        if stats.kstest(data[attr], 'norm')[1] > 0.05:
            p.append(stats.kstest(data[attr], 'norm')[1])
            normal.append('True')
        else:
            p.append(stats.kstest(data[attr], 'norm')[1])
            normal.append('False')
        # plot
        if type == 'pp':
            stats.probplot(data[attr], dist="norm", plot=plt)
            plt.title('Probability Plot of ' + str(attr))
            plt.show()
        else:
            sm.qqplot(data[attr], line='s')
            plt.title('Probability Plot of ' + str(attr))
            plt.show()

    result = pd.DataFrame({'Variable': variable, 'Normal': normal, 'P': p})
    return result


data = pd.read_csv('/Users/martin_yan/Desktop/员工离职预测训练赛/pfm_train.csv')
# data = data[['Age', 'DistanceFromHome', 'Attrition']]
attrs = ['Age', 'DistanceFromHome', 'TotalWorkingYears', 'TrainingTimesLastYear',
         'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole']
check_normality(data, attrs, 'qq')
