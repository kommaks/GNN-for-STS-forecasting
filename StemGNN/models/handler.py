import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os

from utils.math_utils import evaluate


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, data, device, node_cnt, window_size, horizon):
    model.eval()
    target = torch.tensor(data[window_size:]).unsqueeze(0).to(device)
    inputs = torch.tensor(data[:window_size]).unsqueeze(0).float().to(device)
    forecast = torch.empty(1, 0, target.shape[2]).to(device)
    with torch.no_grad():
        for i in range(0, target.shape[1], horizon):
            inputs = inputs[:, -window_size:, :]
            out, _ = model(inputs)
            inputs = torch.cat((inputs, out), dim=1)
            forecast = torch.cat((forecast, out), dim=1)
            
    return forecast.detach().cpu().numpy(), target.detach().cpu().numpy()
            


def validate(model, data, device,
             node_cnt, window_size, horizon):
    forecast, target = inference(model, data, device,
                                           node_cnt, window_size, horizon)
    
    score = evaluate(forecast, target)
    print(f'WMAPE {score[1]:7.9%}; RMSE {score[0]:7.9f}.')
    
    return dict(wmape=score[1], rmse=score[0])


def train(data, args, result_file, cross_val=True, best_score=np.inf):
    node_cnt = data.shape[1]

    

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    performance_metrics = {}
    rmse_total = []
    wmape_total = []
    if cross_val:
        idxs = (np.arange(data.shape[0]-60, 150, -15))
    else:
        idxs = [None]
    
    for idx in idxs:

        model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
        model.to(args.device)
        

        if args.optimizer == 'RMSProp':
            my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
        elif args.optimizer == 'AdamW':
            my_optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        else:
            my_optim = torch.optim.Adamax(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        if cross_val:
            train_data = data[idx - 150: idx]
            valid_data = data[idx - args.window_size: idx + 30]
        else:
            train_data = data[:-30]
            valid_data = data[-(args.window_size + 30):]


        train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon)
    
        train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                            num_workers=0)

        print('------- start of training cycle -------')
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            model.train()
            loss_total = 0
            cnt = 0
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(args.device)
                target = target.to(args.device)
                model.zero_grad()
                forecast, _ = model(inputs)
                loss = forecast_loss(forecast, target)
                cnt += 1
                loss.backward()
                my_optim.step()
                loss_total += float(loss)
            
            if (epoch + 1) % args.validate_freq == 0 and cross_val == False:
                print('------ validate on data: VALIDATE ------')
                performance_metrics = \
                    validate(model, valid_data, args.device,
                            node_cnt, args.window_size, args.horizon)
                
        print('--------- end of training cycle --------')
                
        print('------ validate on data: VALIDATE ------')
        performance_metrics = \
            validate(model, valid_data, args.device,
                    node_cnt, args.window_size, args.horizon)
        wmape, rmse = performance_metrics['wmape'], performance_metrics['rmse']
        wmape_total.append(wmape)
        rmse_total.append(rmse)

    
                
    if cross_val is False and wmape < best_score:
        save_model(model, result_file)


    
            
    return dict(wmape=np.mean(wmape_total), rmse=np.mean(rmse_total))


def test(test_data, args, result_train_file):
    
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    
    performance_metrics = validate(model, test_data, args.device, 
                         node_cnt, args.window_size, args.horizon)
    
    forecast, target = inference(model, test_data, args.device,
                                           node_cnt, args.window_size, args.horizon)
    
    score = evaluate(forecast, target)
    print(f'WMAPE {score[1]:7.9%}; RMSE {score[0]:7.9f}.')
    wmape, rmse = performance_metrics['wmape'], performance_metrics['rmse']
    print('Performance on test set: WMAPE: {:5.4f} | RMSE: {:5.4f}'.format(wmape, rmse))
    return forecast