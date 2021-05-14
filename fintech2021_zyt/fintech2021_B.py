# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:09:14 2021

招商银行fintech2021
    https://www.nowcoder.com/activity/2021cmb/index
初赛B榜单

@author: zhang
"""

###########################################
################ 加载库 ###################
###########################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from functions import *
import warnings
warnings.filterwarnings('ignore')

###########################################
############# 数据读取与处理 ###############
###########################################
dataPath = 'C:/Users/zhang/mine/finTech/2021/data/B'
save_path = 'C:/Users/zhang/mine/finTech/2021/data/B'

trainDf = pd.read_csv(dataPath + '/train_v2.csv')
wkdDf = pd.read_csv(dataPath + '/wkd_v1.csv')
testDf = pd.read_csv(dataPath + '/test_v2_periods.csv')
testDayDf = pd.read_csv(dataPath + '/test_v2_day.csv')

testDf_original = testDf.copy()
testDayDf_original = testDayDf.copy()  
wkdDf = wkdDf.rename(columns={'ORIG_DT':'date'})

######### 数据合并：日期类型信息
######### 这里数据官方已清洗过
trainDf = trainDf.merge(wkdDf, on = ['date'], how = 'left')
testDf = testDf.merge(wkdDf, on = ['date'], how = 'left')
testDayDf = testDayDf.merge(wkdDf, on = ['date'], how = 'left')

## 提取岗位以日为粒度的业务量
trainDayDf = get_jobDayAmount(trainDf)
## 提取岗位以0.5小时为粒度的业务量
trainDf = get_jobPeriodAmount(trainDf)

## 加上岗位的业务量
trainDayDf['bizCount'] = trainDayDf['post_id'].apply(lambda x: 13 if x == 'A' else 1)
testDayDf['bizCount'] = testDayDf['post_id'].apply(lambda x: 13 if x == 'A' else 1)

## 加上是否工作日
wkdDict = {'WN':0, 'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0}
trainDf['isWKD'] = trainDf['WKD_TYP_CD'].map(wkdDict)
testDf['isWKD'] = testDf['WKD_TYP_CD'].map(wkdDict)

## 提取时间特征，此处以年、月、日作为变量
########## 
## ['post_id', 'periods', 'WKD_TYP_CD', 'amount', 'year', 'month', 'day']
trainDf = getDateDf(trainDf)
trainDayDf = getDateDf(trainDayDf)
testDf = getDateDf(testDf)
testDayDf = getDateDf(testDayDf)

trainDayDf = trainDayDf[~((trainDayDf['year'] == 2019)&(trainDayDf['month'] == 12)&(trainDayDf['day'] == 30))]

###########################################
################ 模型部分 ##################
###########################################
###### 这里只简单地采用了随机森林

#########################################
################ task1 ##################
#########################################

#####################################
############## day ##################
#####################################
#### 以day为粒度进行预测分析
#### 对字符向量进行labelEncoder
trainDayCols = trainDayDf.columns.tolist()
testDayCols = testDayDf.columns.tolist()

trainDayDf['isTest'] = -1
testDayDf['isTest'] = 1
totalDayDf = pd.concat([trainDayDf, testDayDf])
cols = ['post_id', 'WKD_TYP_CD']
for col in cols:
    if totalDayDf[col].dtype == 'object':
        totalDayDf[col] = totalDayDf[col].astype(str)
labelEncoder_df(totalDayDf, cols)

trainDayDf = totalDayDf[totalDayDf['isTest'] == -1]
trainDayDf = trainDayDf[trainDayCols]
testDayDf = totalDayDf[totalDayDf['isTest'] == 1]
testDayDf = testDayDf[testDayCols]
trainDayDf['amount'] = trainDayDf['amount'].astype(int)

## 降内存
del totalDayDf
trainDayDf = reduce_mem_usage(trainDayDf)
testDayDf = reduce_mem_usage(testDayDf)

######## 模型训练
trainDayX = trainDayDf.drop(['amount'], axis = 1)
trainDayY = trainDayDf['amount']

## 以0.5h为粒度的task2
random_seed = 2021
np.random.seed(2)
    
rf_cflDay = RandomForestRegressor(n_estimators = 250,
                                  random_state = 2021, # 42
                               )
rf_cflDay.fit(trainDayX, trainDayY)
print("Train Score:%f" % rf_cflDay.score(trainDayX, trainDayY))

testDayX = testDayDf.drop(['amount'], axis = 1)
y_predict_rfDay = rf_cflDay.predict(testDayX)
y_predict_rfDay = y_predict_rfDay.astype(int)

testDayDf_original['amount'] = y_predict_rfDay.tolist()
testDayDf_original.to_csv(save_path + '/test_Day_rf.txt', sep = ',', header = True, index = False, encoding = 'utf-8')


#########################################
################ task2 ##################
#########################################

######################################
############## 0.5h ##################
######################################
#### 以0.5h为粒度进行预测分析
#### 对字符向量进行labelEncoder
trainCols = trainDf.columns.tolist()
testCols = testDf.columns.tolist()

trainDf['isTest'] = -1
testDf['isTest'] = 1
totalDf = pd.concat([trainDf, testDf])
cols = ['post_id', 'WKD_TYP_CD']
for col in cols:
    if totalDf[col].dtype == 'object':
        totalDf[col] = totalDf[col].astype(str)
labelEncoder_df(totalDf, cols)

trainDf = totalDf[totalDf['isTest'] == -1]
trainDf = trainDf[trainCols]
testDf = totalDf[totalDf['isTest'] == 1]
testDf = testDf[testCols]
trainDf['amount'] = trainDf['amount'].astype(int)

## 降内存
del totalDf
trainDf = reduce_mem_usage(trainDf)
testDf = reduce_mem_usage(testDf)

######## 模型训练
trainX = trainDf.drop(['amount'], axis = 1)
trainY = trainDf['amount']

## 以0.5h为粒度的task2
random_seed = 2021
np.random.seed(2)
    
rf_cfl = RandomForestRegressor(n_estimators = 300,
                               random_state = 2021, # 2021
                               )
rf_cfl.fit(trainX, trainY)
print("Train Score:%f" % rf_cfl.score(trainX, trainY))

testX = testDf.drop(['amount'], axis = 1)
y_predict_rf = rf_cfl.predict(testX)
y_predict_rf = y_predict_rf.astype(int)

testDf_original['amount'] = y_predict_rf.tolist()
testDf_original.to_csv(save_path + '/test_period_rf.txt', sep = ',', header = True, index = False, encoding = 'utf-8')
