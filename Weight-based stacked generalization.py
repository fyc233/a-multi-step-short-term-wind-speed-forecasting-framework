#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from test_model.model.ModelFactory_point import SWLSTMmodel, MGMmodel, LSTMmodel, GRUmodel, SSWGMmodel, SiTGRUmodel
from test_model.evaluation.Evaluation import Evaluation
from test_model.dataoperator.series_to_supervised import data_split
import scipy.io as io
import warnings
warnings.filterwarnings("ignore")

# In[21]:
#分解后每个子序列分成506*13输入矩阵与506*1输出矩阵,输入矩阵分为354*13训练集与152*13测试集,
#交叉验证将训练集分成283*13训练集与71*13验证集(该过程是将354分成5组循环5次，每组当一次验证集)

#整个过程均在分解结束后，对子序列PSR之后得到的输入输出矩阵,以下为将其划分为训练集测试集过程
# x=load_boston().data               #输入矩阵 矩阵506*13
# y=load_boston().target             #输出矩阵 矩阵506*1
x1=pd.read_excel(r"F:\shuju.xls",sheet_name="Sheet1")
y1=pd.read_excel(r"F:\shuju.xls",sheet_name="Sheet2")

x1=MinMaxScaler().fit_transform(x1)  #输入矩阵归一化 范围(0,1)mm = MinMaxScaler()
# nn = MinMaxScaler()
# y1=nn.fit_transform(y1)
mm = MinMaxScaler()
y1 = mm.fit_transform(y1)
y1 = mm.inverse_transform(y1)
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=100) #x,y均取70%为训练集并将其重复执行100次
# kf=KFold(n_splits=5,random_state=100) #交叉验证，分成五组其中四组作为训练一组作为验证,每组71个数
# n_train=xtrain.shape[0]         #354个训练样本
# n_test=xtest.shape[0]           #152个测试样本
# n_train
# In[22]:
#2 4 5 6
# models=[SVR(kernel="rbf"),SVR(kernel="linear"),RandomForestRegressor(n_estimators=300,random_state=100),GradientBoostingRegressor(n_estimators=300,random_state=100),XGBRegressor(n_estimators=300),ExtraTreesRegressor(n_estimators=300,n_jobs=-1,random_state=100),RandomForestRegressor(n_estimators=300,random_state=100),GradientBoostingRegressor(n_estimators=300,random_state=100),LGBMRegressor(n_estimators=300,n_jobs=-1,random_state=100)]
#models=[GradientBoostingRegressor(n_estimators=300,random_state=100),XGBRegressor(n_estimators=300),ExtraTreesRegressor(n_estimators=300,n_jobs=-1,random_state=100),RandomForestRegressor(n_estimators=300,random_state=100)]
models=[GradientBoostingRegressor(n_estimators=300,random_state=100),ExtraTreesRegressor(n_estimators=300,n_jobs=-1,random_state=100),RandomForestRegressor(n_estimators=300,random_state=100)]
# 极端随机树回归 XGBoost 随机数回归 GBDT回归
#models=[RandomForestRegressor(n_estimators=300,random_state=100),GradientBoostingRegressor(n_estimators=300,random_state=100),LGBMRegressor(n_estimators=300,n_jobs=-1,random_state=100),RidgeCV(alphas=[0.0001,0.001,0.01,0.1,0.2,0.5,1,2,3,4,5,10,20,30,50]),LinearRegression(),SVR(kernel="rbf"),SVR(kernel="linear"),RandomForestRegressor(n_estimators=300,random_state=100),GradientBoostingRegressor(n_estimators=300,random_state=100),XGBRegressor(n_estimators=300),ExtraTreesRegressor(n_estimators=300,n_jobs=-1,random_state=100)]
#包含11个模型 RandomForestRegressor随机森林回归：n_estimators种群数量,random_state保证每次预测结果相同;
#           GradientBoostingRegressor梯度增强回归;LGBMRegressor
#           RidgeCV内置交叉验证:alphas正则化程度,越大程度越强,均浮点数

def improved_stacking_oof_test(x_train,y_train,oof_train,oof_test_skf):
    # 求一个模型内五次交叉验证中验证集oof_train与y_train比较误差
    # 求出五个误差后计算各自权重
    # 根据权重代入oof_test_skf求oof_test
    evaluation = Evaluation()
    oof_wucha=np.zeros((Kfold_number,)) #根据交叉次数改
    oof_weight=np.zeros((Kfold_number,)) #根据交叉次数改
    oof_test = np.zeros((n_test,))
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        kf_oof_train = oof_train[test_index]
        kf_y_train = y_train[test_index]
        pointMetrics = evaluation.getPointPredictionMetric(kf_oof_train, kf_y_train)
        oof_wucha[i]=pointMetrics["RMSE"]
    a=np.sum(oof_wucha)
    for i in range(Kfold_number): #根据交叉次数改
        oof_weight[i]=(a-oof_wucha[i])/(a*(Kfold_number-1)) #根据交叉次数改
        oof_test[:]=oof_test[:]+oof_test_skf[i,:]*oof_weight[i]
    return oof_test

def get_oof(model,x_train,y_train,x_test):
    oof_train=np.zeros((n_train,))      #构造一个1*354的一维0矩阵
    oof_test=np.zeros((n_test,))        #构造一个1*152的一维0矩阵
    #oof_test1 = np.zeros((n_test,))  # 构造一个1*152的一维0矩阵
    oof_test = np.zeros((n_test,))  # 构造一个1*152的一维0矩阵
    oof_test_skf=np.zeros((Kfold_number,n_test))   #5*152(5为交叉次数五次) 根据交叉次数改
    oof_wucha = np.zeros((Kfold_number,))
    for i,(train_index,test_index) in enumerate(kf.split(x_train)): #意义见https://blog.csdn.net/HiSi_/article/details/108127173
        kf_x_train=x_train[train_index] #交叉验证第一层循环将后四组共283*13设为训练集,后面循环一次类推
        kf_x_test=x_train[test_index]   #交叉验证第一层循环将前一组设为验证集,后面循环一次类推
        kf_y_train=y_train[train_index] #同理设置y_train 283个值
        model=model.fit(kf_x_train,kf_y_train) #拿kf_x_train,kf_y_train训练预测模型
        oof_train[test_index]=model.predict(kf_x_test)#训练出的模型预测第一组验证集,每次产生71个预测值,最终5折后成为堆叠成为1*354个训练样本的测试值
        oof_test_skf[i,:]=model.predict(x_test)       #训练出的模型代入输入测试集矩阵预测输出测试集矩阵,每次生成1*152的测试集预测值，填oof_test_skf[i，：]，五次以后填满形成5*152的预测值矩阵
    #oof_test[:]=oof_test_skf.mean(axis=0)            #把测试集的五次预测结果，求平均，形成一次预测结果
    oof_test[:]=improved_stacking_oof_test(x_train,y_train,oof_train,oof_test_skf)
    return oof_train,oof_test     #第一个返回值为第二层模型xtrain的特征，1*354；第二个返回值为第一层模型对测试集数据的预测1*152，要作为第二层模型的训练集Xtest

# void(main)
K=12 #分解次数
dim=10 #PSR参数
Kfold_number=3
modelsnumber=3
n = x1.shape[1]
nn = x1.shape[0]
x=x1[:, 0:dim]
y=y1[:, 0]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)  # x,y均取70%为训练集并将其重复执行100次
n_test = 297
# xtrain, xtest = data_split(x, n_test, 0)
# ytrain, ytest = data_split(y, n_test, 0)
n_test=xtest.shape[0]           #152个测试样本
number_models = len(models)
predictions=np.zeros((n_test,K))
predictions_test=np.zeros((n_test,1))
realize=np.zeros((n_test,K))
prediction_realize=np.zeros((n_test,1))
n_train=xtrain.shape[0]
xtrain_new_mat=np.zeros([n_train,number_models*K])
xtest_new_mat=np.zeros([n_test,number_models*K])
ytrain_mat=np.zeros([n_train,K])
ytest_mat=np.zeros([n_test,K])
x_gai_new=np.zeros([x.shape[0],number_models*K])
y_gai_new=np.zeros([y.shape[0],K])
xtrain_new = np.zeros((n_train, number_models))  # 设置354*11空矩阵
xtest_new = np.zeros((n_test, number_models))  # 512*11空矩阵
for i in range(0,K,1): # 5是由于PSR中dim=5
    x=x1[:, i*dim:i*dim+dim]
    y=y1[:, i]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)  # x,y均取70%为训练集并将其重复执行100次
    # xtrain, xtest = data_split(x, n_test, 0)
    # ytrain, ytest = data_split(y, n_test, 0)
    kf=KFold(n_splits=Kfold_number,random_state=100) #交叉验证，分成五组其中四组作为训练一组作为验证,每组71个数
    n_train=xtrain.shape[0]         #354个训练样本
    n_test=xtest.shape[0]           #152个测试样本
    for j, regressor in enumerate(models):
        xtrain_new[:, j], xtest_new[:, j] = get_oof(regressor, xtrain, ytrain, xtest)
    # for ii in range(0,x.shape[0],1):
    #     x_realize=x[ii,0]
    #     for jj in range(0,xtrain.shape[0],1):
    #         x_gai=xtrain[jj,0]
    #         if x_realize==x_gai:
    #             x_gai_new[ii]=xtrain_new[jj]
    #     for jj in range(0,xtest.shape[0],1):
    #         x_gai=xtest[jj,0]
    #         if x_realize==x_gai:
    #             x_gai_new[ii]=xtest_new[jj]
    # for ii in range(0,y.shape[0],1):
    #     y_realize=y[ii]
    #     for jj in range(0,ytrain.shape[0],1):
    #         y_gai=ytrain[jj]
    #         if y_realize==y_gai:
    #             y_gai_new[ii]=ytrain[jj]
    #     for jj in range(0,ytest.shape[0],1):
    #         y_gai=ytest[jj]
    #         if y_realize==y_gai:
    #             y_gai_new[ii]=ytest[jj]
    # xtrain_new, xtest_new = data_split(x_gai_new, n_test, 0)
    # ytrain, ytest = data_split(y_gai_new, n_test, 0)
    if i == 0:
        xtrain_new_mat[:, :number_models] = xtrain_new[:, :number_models]
        xtest_new_mat[:, :number_models] = xtest_new[:, :number_models]
    else:
        xtrain_new_mat[:, i * modelsnumber:i * modelsnumber + modelsnumber] = xtrain_new[:, :number_models]
        xtest_new_mat[:, i * modelsnumber:i * modelsnumber + modelsnumber] = xtest_new[:, :number_models]
    ytrain_mat[:,i] = ytrain
    ytest_mat[:,i] = ytest
    reg = LinearRegression()
    reg = reg.fit(xtrain_new, ytrain)  # 新x_train与旧y_train训练预测模型
    score = reg.score(xtest_new, ytest)  # 训练好的预测模型代入新x_test预测y_test_new,预测结果与旧y_test比较
    predictions[:,i] = reg.predict(xtest_new)
    realize[:,i]=ytest

ytrain_mat=ytrain_mat.swapaxes(0, 1)
ytest_mat=ytest_mat.swapaxes(0, 1)
for ii in range(0,x.shape[0],1):
    x_realize=x[ii,0]
    for jj in range(0,xtrain.shape[0],1):
        x_gai=xtrain[jj,0]
        if x_realize==x_gai:
            x_gai_new[ii]=xtrain_new_mat[jj]
    for jj in range(0,xtest.shape[0],1):
        x_gai=xtest[jj,0]
        if x_realize==x_gai:
            x_gai_new[ii]=xtest_new_mat[jj]
ytrain_mat=np.transpose(ytrain_mat)
ytest_mat=np.transpose(ytest_mat)
for ii in range(0,y.shape[0],1):
    y_realize=y[ii]
    for jj in range(0,ytrain.shape[0],1):
        y_gai=ytrain[jj]
        if y_realize==y_gai:
            y_gai_new[ii]=ytrain_mat[jj]
    for jj in range(0,ytest.shape[0],1):
        y_gai=ytest[jj]
        if y_realize==y_gai:
            y_gai_new[ii]=ytest_mat[jj]
xtrain_new_mat, xtest_new_mat = data_split(x_gai_new, n_test, 0)
# ytrain_mat, ytest_mat = data_split(y_gai_new, n_test, 0)
# prediction_realize = realize.sum(axis=1)
# prediction_realize=np.sum(realize, axis=0)
# predictions_test= predictions.sum(axis=1)
# evaluation = Evaluation()
# pointMetrics = evaluation.getPointPredictionMetric(predictions_test,prediction_realize)
# print(pointMetrics)

mat_path = r'F:\ytrain.mat'
io.savemat(mat_path, {'name': ytrain_mat})
mat_path = r'F:\ytest.mat'
io.savemat(mat_path, {'name': ytest_mat})
mat_path = r'F:\xtrain.mat'
io.savemat(mat_path, {'name': xtrain_new_mat})
mat_path = r'F:\xtest.mat'
io.savemat(mat_path, {'name': xtest_new_mat})
# In[23]:
# #第二层
# xtrain_new可作为输入矩阵强预测模型训练集,xtest_new测试集
# reg=LinearRegression()
# reg=reg.fit(xtrain_new,ytrain) #新x_train与旧y_train训练预测模型
# score=reg.score(xtest_new,ytest)#训练好的预测模型代入新x_test预测y_test_new,预测结果与旧y_test比较
# predictions = reg.predict(xtest_new)
# evaluation = Evaluation()
# pointMetrics = evaluation.getPointPredictionMetric(predictions, ytest)
# print(pointMetrics)
# metric = np.array(
#             [pointMetrics['MAE'], pointMetrics['RMSE'], pointMetrics['MAPE']])


# # In[ ]:
xtrain_new_mat = xtrain_new_mat.reshape((xtrain_new_mat.shape[0], xtrain_new_mat.shape[1], 1))  # 将2维数组变成3维的
xtest_new_mat = xtest_new.reshape((xtest_new_mat.shape[0], xtest_new_mat.shape[1], 1))

n_input = 9
n_nodes = 40  # The number of LSTM units to use in the hidden layer
n_epochs = 100
n_batch = 32
n_features = 1
ilr = 0.04    #学习率
predictions=[]
metrics = []
config = [n_input, n_nodes, n_epochs, n_batch, n_features, ilr]

model = LSTMmodel(xtrain_new_mat, ytrain, config)
predictions = model.predict(xm_test, verbose=1)

evaluation = Evaluation()
pointMetrics = evaluation.getPointPredictionMetric(predictions, ytest)
print(pointMetrics)
metric = np.array(
            [pointMetrics['MAE'], pointMetrics['RMSE'], pointMetrics['MAPE']])
# # In[ ]:
#
#
#
#
