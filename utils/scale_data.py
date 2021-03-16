#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File : scale_data.py
# @Author : Martin Yan
# @Time : 2021/3/16 下午7:19
# @Software : PyCharm

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def minmax_scale(data, attrs):
    """
    Use MinMaxScaler function to scale data in range default=(0, 1)
    :param data: origin data
    :param attrs: list of attributes
    :return: scaled data
    """
    for attr in attrs:
        col = data[attr]
        data.drop(attr, axis=1, inplace=True)
        scaled = MinMaxScaler().fit_transform(col.values.reshape(-1, 1))
        temp = pd.DataFrame(scaled, columns=[attr])
        data.index = temp.index
        data = pd.concat([data, temp], axis=1)
    return data


def robust_scale(data, attrs):
    """
    Use RobustScaler function to scale data that are robust to outliers
    :param data: origin data
    :param attrs: list of attributes
    :return: scaled data
    """
    for attr in attrs:
        col = data[attr]
        data.drop(attr, axis=1, inplace=True)
        scaled = RobustScaler().fit_transform(col.values.reshape(-1, 1))
        temp = pd.DataFrame(scaled, columns=[attr])
        data.index = temp.index
        data = pd.concat([data, temp], axis=1)
    return data


def standardize_scale(data, attrs):
    """
    Use StandardScaler function to scale data by removing the mean and scaling to unit variance
    :param data: origin data
    :param attrs: list of attributes
    :return: scaled data
    """
    for attr in attrs:
        col = data[attr]
        data.drop(attr, axis=1, inplace=True)
        scaled = StandardScaler().fit_transform(col.values.reshape(-1, 1))
        temp = pd.DataFrame(scaled, columns=[attr])
        data.index = temp.index
        data = pd.concat([data, temp], axis=1)
    return data

