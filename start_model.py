import torch
import numpy as np
import train_functions
from models.gcn import GCN
from models.gru import GRU 
from models.tgcn import TGCN
from models.a3t_gcn import TGCNAttention

def initialization(model_name, trainX, trainY, testX, testY, adj, learning_rate, gru_units, seq_len, pre_len, training_epoch, totalbatch, batch_size, num_nodes, device):
    mse_loss = torch.nn.MSELoss()
    if model_name == 'TGCN':
        TGCN_module = TGCN(adj=adj, hidden_dim=gru_units).to(device)
        optimizer = torch.optim.AdamW(params=TGCN_module.parameters(), lr=learning_rate)
        print('Number of weights:', np.sum([np.prod(p.shape) for p in TGCN_module.parameters()]))
        y_pred, test_rmse, test_wmape = train_functions.train(trainX, trainY, testX, testY, 
                                                            clf_nn=TGCN_module, 
                                                            criterion=mse_loss, 
                                                            optimizer=optimizer,
                                                            training_epoch=training_epoch,
                                                            totalbatch=totalbatch,
                                                            batch_size=batch_size,
                                                            num_nodes=num_nodes,
                                                            device=device)

    if model_name == 'GCN':
        GCN_module = GCN(adj=adj, input_dim=seq_len, output_dim=gru_units).to(device)
        optimizer = torch.optim.AdamW(params=GCN_module.parameters(), lr=learning_rate)
        print('Number of weights:', np.sum([np.prod(p.shape) for p in GCN_module.parameters()]))
        y_pred, test_rmse, test_wmape = train_functions.train(trainX, trainY, testX, testY, 
                                                            clf_nn=GCN_module,
                                                            criterion=mse_loss, 
                                                            optimizer=optimizer,
                                                            training_epoch=training_epoch,
                                                            totalbatch=totalbatch,
                                                            batch_size=batch_size,
                                                            num_nodes=num_nodes,
                                                            device=device)


    if model_name == 'GRU':
        GRU_module = GRU(input_dim=num_nodes, hidden_dim=gru_units).to(device)
        optimizer = torch.optim.AdamW(params=GRU_module.parameters(), lr=learning_rate)
        print('Number of weights:', np.sum([np.prod(p.shape) for p in GRU_module.parameters()]))
        y_pred, test_rmse, test_wmape = train_functions.train(trainX, trainY, testX, testY, 
                                                            clf_nn=GRU_module,
                                                            criterion=mse_loss, 
                                                            optimizer=optimizer,
                                                            training_epoch=training_epoch,
                                                            totalbatch=totalbatch,
                                                            batch_size=batch_size,
                                                            num_nodes=num_nodes,
                                                            device=device)                                                          

    if model_name == 'A3T-GCN':
        TGCNAttention_module = TGCNAttention(adj=adj, seq_len=seq_len, pre_len=pre_len, hidden_dim=gru_units).to(device)
        optimizer = torch.optim.AdamW(params=TGCNAttention_module.parameters(), lr=learning_rate)
        print('Number of weights:', np.sum([np.prod(p.shape) for p in TGCNAttention_module.parameters()]))
        y_pred, test_rmse, test_wmape = train_functions.train(trainX, trainY, testX, testY, 
                                                            clf_nn=TGCNAttention_module,
                                                            criterion=mse_loss, 
                                                            optimizer=optimizer,
                                                            training_epoch=training_epoch,
                                                            totalbatch=totalbatch,
                                                            batch_size=batch_size,
                                                            num_nodes=num_nodes,
                                                            device=device)
    
    return y_pred, test_rmse, test_wmape

