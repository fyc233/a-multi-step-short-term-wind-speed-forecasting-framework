#####BiDLSTM######
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler


def generator(data, lag_step, delay_step):
    factors = data[:, 0]
    label = data[:, 0]
    x = []
    for i in range(len(factors) - lag_step+1-delay_step):
        x.append(factors[i:(i + lag_step)])
    y = []
    for i in range(lag_step, (len(label)+1-delay_step)):
        y.append(label[i:i+delay_step])
    x = np.array(x)
    y = np.array(y)
    return x, y

import scipy.io as io
# load data
data = np.loadtxt('2021.2.11.txt')

# normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data.reshape(-1, 1))

# reconstruct phase space
embedding_dim = 10  # embedding dimension
delay = 5  # time delay

# split into training and testing sets
train_size = int(0.7 * len(data_norm))
train_X = data_norm[:train_size]
test_X = data_norm[int(0.8 * len(data_norm)):]
train_X, train_y = generator(train_X, embedding_dim, delay)
test_X, test_y = generator(test_X, embedding_dim, delay)

# reshape input to be 3D [samples, timesteps, features]
train_X = np.reshape(train_X, (train_X.shape[0], embedding_dim, 1))
test_X = np.reshape(test_X, (test_X.shape[0], embedding_dim, 1))


# build BID-LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv1D, GRU, Flatten
from sklearn.metrics import mean_squared_error as mse
# from test_model.evaluation.Evaluation import Evaluation
import numpy as np




model = Sequential()
# model.add(Conv1D(filters=20, kernel_size=3, strides=1, padding='causal',  activation='relu'))
# model.add(Bidirectional(LSTM(20, activation='relu')))
# model.add(GRU(20, activation='relu', input_shape=(embedding_dim, 1)))
model.add(Dense(20, activation='relu', input_shape=(embedding_dim, 1)))
model.add(Flatten())
model.add(Dense(delay))
model.compile(optimizer='adam', loss='mse')

# train the model
# model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=2)
model.fit(train_X, train_y, epochs=20, batch_size=16, verbose=1)
# make predictions
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

# invert predictions to original scale
train_pred_inv = scaler.inverse_transform(train_pred)
train_y_inv = scaler.inverse_transform(train_y)
test_pred_inv = scaler.inverse_transform(test_pred)
test_y_inv = scaler.inverse_transform(test_y)
L, D = test_y_inv.shape[0], test_y_inv.shape[1]
test_pred_inv = test_pred_inv.reshape(-1, 1)
test_y_inv = test_y_inv.reshape(-1, 1)
# plot predictions
import matplotlib.pyplot as plt
# mat_path = r'F:\BiLSTM.mat'
# io.savemat(mat_path, {'name': test_pred_inv})
# mat_path = r'F:\ytest_BiLSTM.mat'
# io.savemat(mat_path, {'name': test_y_inv})
# plt.plot(data, label='True')
# # plt.plot(np.concatenate([train_pred_inv, test_pred_inv]), label='Predicted')
# plt.legend()
# plt.show()
# evaluation = Evaluation()
# pointMetrics = evaluation.getPointPredictionMetric(test_pred_inv,test_y_inv)

print('rmse', math.sqrt(mse(test_pred_inv,test_y_inv)))
import pandas as pd
test_y_inv = test_y_inv.reshape(L, D)
test_pred_inv = test_pred_inv.reshape(L, D)
# # 一维列向量数据
# data = test_pred_inv
#
# # 创建 DataFrame，将数据放在第一列
# df = pd.DataFrame(data, columns=['Column1'])
#
# # 创建 ExcelWriter 对象
# writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
#
# # 写入第二个工作表
# df.iloc[:, 0].to_frame(name='Column3').to_excel(writer, sheet_name='Sheet2', columns=['C'], index=False)
#
# # 关闭 ExcelWriter 对象
# writer.save()
# import pandas as pd

# 创建 DataFrame，将数据放在第一列

df = pd.DataFrame(test_pred_inv)

# 创建 ExcelWriter 对象
writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

# 写入数据到第一个工作表
df.to_excel(writer, sheet_name='Sheet1', index=False)

# 关闭 ExcelWriter 对象
writer.save()

from test_model.evaluation.Evaluation import Evaluation
evaluation = Evaluation()
pointMetrics = evaluation.getPointPredictionMetric(test_pred_inv, test_y_inv)
print(pointMetrics)
metric = np.array(
            [pointMetrics['RMSE'], pointMetrics['MAE'], pointMetrics['MAPE'],pointMetrics['NSE'], pointMetrics['MSE']])