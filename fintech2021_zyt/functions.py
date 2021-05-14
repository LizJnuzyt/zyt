# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:55:12 2021

招商银行fintech2021
    https://www.nowcoder.com/activity/2021cmb/index
函数部分

@author: zhang
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

###########################################
"""
数据处理部分函数
"""
###########################################

## 提取岗位以日为粒度的业务量
def get_jobDayAmount(df):
    outputDf = df.groupby(['date', 'post_id', 'WKD_TYP_CD'], as_index = False)['amount'].sum()
    #outputDf = outputDf.sort_values(by = ['date', 'post_id'], axis = 0, ascending = True)
    return outputDf

## 提取细分业务岗位以日为粒度的业务量
def get_jobBizDayAmount(df):
    outputDf = df.groupby(['date', 'post_id', 'biz_type', 'WKD_TYP_CD'], as_index = False)['amount'].sum()
    #outputDf = outputDf.sort_values(by = ['date', 'post_id'], axis = 0, ascending = True)
    return outputDf


## 提取岗位以0.5小时为粒度的业务量
def get_jobPeriodAmount(df):
    outputDf = df.groupby(['date', 'post_id', 'periods', 'WKD_TYP_CD'], as_index = False)['amount'].sum()
    #outputDf = outputDf.sort_values(by = ['date', 'post_id', 'periods'], axis = 0, ascending = True)
    return outputDf

## 提取时间特征，此处以年、月、日作为变量
def getDateDf(df):
    df['date'] = pd.to_datetime(df['date'], format = '%Y/%m/%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(['date'], axis=1, inplace=True)
    return df

## 将日期还原
def repairDate(df):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['date'] = df['date'].apply(lambda x: datetime.strftime(x, format = '%Y/%#m/%#d'))
    df.drop(['year', 'month', 'day'], axis = 1, inplace = True)
    return df

###########################################
"""
模型相关部分函数
"""
###########################################
## 将字符向量转化为值向量
def labelEncoder_df(df, features):
    for i in features:
        encoder = LabelEncoder()
        df[i] = encoder.fit_transform(df[i])
        
## 降低内存
## 借鉴狼崽
# https://github.com/wolfkin-hth/FinTech2020/blob/master/Final_code/method.py
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df