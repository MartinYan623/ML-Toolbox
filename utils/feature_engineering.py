#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File : feature_engineering.py
# @Author : Martin Yan
# @Time : 2021/3/16 下午7:46
# @Software : PyCharm

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder


def one_hot_encoding(data, attrs):
    """
    Convert categorical attributes to a new binary variable
    :param data: origin data
    :param attrs: list of attributes
    :return: one-hot data
    """
    for attr in attrs:
        temp = pd.get_dummies(data[attr], prefix=attr)
        data.index = temp.index
        data = pd.concat([data, temp], axis=1)
        data.drop(attr, axis=1, inplace=True)
    return data


def label_encoding(data, attrs):
    """
    Encode target labels with value between 0 and n_classes-1
    :param data: origin data
    :param attrs: list of attributes
    :return: labeled data
    """
    for attr in attrs:
        le = LabelEncoder()
        temp = le.fit_transform(data[attr])
        data[attr] = temp
    return data


def cut_bin(data, attr, bins, labels):
    """
    Cut values of the attribute automatically to new values
    :param data: origin data
    :param attr: a attribute
    :param bin: the number of bins
    :param labels: new values of the attribute
    :return: discretization data
    """
    data[attr] = pd.cut(data[attr], bins, labels=labels)
    return data


def kbins_discretizer(data, attr, bins, encode='ordinal', strategy='quantile'):
    """
    Bin continuous data into intervals
    :param data: origin data
    :param attr: a attribute
    :param bins: the number of bins
    :param encode: method used to encode the transformed result {‘onehot’, ‘onehot-dense’, ‘ordinal’}, default=’onehot’
    :param strategy: strategy used to define the widths of the bins {‘uniform’, ‘quantile’, ‘kmeans’}, default=’quantile’
    :return: discretization data
    """
    trans = KBinsDiscretizer(n_bins=bins, encode=encode, strategy=strategy)
    temp = np.array(data[attr]).reshape(-1, 1)
    temp_trans = trans.fit_transform(temp)
    data[attr] = temp_trans.reshape(-1).astype('int')
    return data


# data = pd.read_csv('/Users/martin_yan/Desktop/员工离职预测训练赛/pfm_train.csv')
# data = kbins_discretizer(data, 'Age', 5)
# print(data['Age'])
