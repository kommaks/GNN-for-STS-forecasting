import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_files(file_path, dtype=np.float32):
    file_df = pd.read_csv(file_path, header=None)
    file = np.array(file_df, dtype=dtype)
    return file

def load_files_prophet(file_path):
    file_df = pd.read_csv(file_path)
    return file_df

## Evaluation
def evaluation(y,yhat):
    y = y.detach().cpu().numpy()
    yhat = yhat.detach().cpu().numpy()

    mse = mean_squared_error(y, yhat, squared=True)
    rmse = mean_squared_error(y, yhat, squared=False)
    mae = mean_absolute_error(y, yhat)
    wmape_score = np.abs(y - yhat).sum() / np.abs(y).sum()
    mape_score = np.mean(np.abs((y - yhat)) / y)

    return rmse, mae, wmape_score, mape_score

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size - seq_len : time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])

    testX.append(data1[train_size - seq_len : train_size])
    testY.append(data1[train_size : time_len])
    return trainX, trainY, testX, testY

def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian