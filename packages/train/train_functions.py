import numpy as np
import torch
from packages.utils.utils import evaluation

def epoch_train(inputs, labels, clf, optimizer, criterion, num_nodes):

    ## Parameters
    label = torch.reshape(labels, [-1,num_nodes])

    ## Prediction
    y_preds = clf.forward(inputs)
    y_pred = torch.reshape(y_preds, [-1,num_nodes])

    ## loss
    loss = criterion(y_pred, label)

    ## RMSE
    error = torch.sqrt(torch.mean(torch.square(y_pred-label)))

    ## Optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, error, y_pred

def epoch_test(inputs, labels, clf, criterion, num_nodes):

    ## Parameters
    label = torch.reshape(labels, [-1,num_nodes])

    ## Prediction
    y_preds = clf.forward(inputs)
    y_pred = torch.reshape(y_preds, [-1,num_nodes])

    ## loss
    loss = criterion(y_pred, label)

    ## RMSE
    error = torch.sqrt(torch.mean(torch.square(y_pred-label)))

    return loss, error, y_pred

def train(trainX, trainY, testX, testY, clf_nn, criterion, optimizer, training_epoch, totalbatch, batch_size, num_nodes, device):
    test_rmse, test_wmape = [], []
    testX, testY = torch.from_numpy(np.array(testX)).to(device), torch.from_numpy(np.array(testY)).to(device)

    for epoch in range(training_epoch):
        for m in range(totalbatch):
            mini_batch = trainX[m * batch_size : (m+1) * batch_size]
            mini_label = trainY[m * batch_size : (m+1) * batch_size]

            inputs = torch.tensor(np.array(mini_batch)).to(device)
            labels = torch.tensor(np.array(mini_label)).to(device)

            loss1, rmse1, train_output = epoch_train(inputs, labels, clf_nn, optimizer, criterion, num_nodes)

        # Test completely at every epoch
        loss2, rmse2, test_output = epoch_test(inputs=testX, labels=testY, clf=clf_nn, criterion=criterion, num_nodes=num_nodes)

        test_label = torch.reshape(testY,[-1,num_nodes])

        rmse, mae, wmape_score, mape_score = evaluation(test_label, test_output)
        test_rmse.append(rmse)
        test_wmape.append(wmape_score)

        print('Iter: {}'.format(epoch+1),
            'mae: {:.4}'.format(mae),
            'rmse: {:.4}'.format(rmse),
            'mape: {:.4}'.format(mape_score),
            'wmape_score: {:.4}'.format(wmape_score))
    

    return test_output, test_rmse, test_wmape