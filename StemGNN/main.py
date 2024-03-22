import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
import pandas as pd
import numpy as np

import optuna

import warnings
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--cross_val', type=bool, default=False)
parser.add_argument('--n_retrains', type=int, default=30)
parser.add_argument('--dataset', type=str, default='atm_transactions')
parser.add_argument('--window_size', type=int, default=30)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=1.09e-3)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.166)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--leakyrelu_rate', type=int, default=0.637)



args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')
result_train_file = os.path.join('model_weights')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists('forecast'):
    os.makedirs('forecast')
data = pd.read_csv(data_file).values
data = np.array(data)[:, 1:]



def normalize(data):
    data = data[:, np.argwhere(data.sum(axis=0) > 1000.0).reshape(-1)]
    scale = data.max(axis=0).reshape(1, -1)
    data = data / scale
    return data, scale

def de_normalize(data):
    data = data * scale
    return data

data, scale = normalize(data)


torch.manual_seed(0)
specific_params = {
    'window_size': ('cat', [7, 14, 30, 40]),
    'horizon': ('cat', [5, 10, 30]),
    'lr': ('float', [1e-4, 1e-2]),
    'multi_layer': ('int', [5, 10]),
    'epoch': ('int', [70, 100]),
    'decay_rate': ('float', [0.01, 0.99]),
    'leakyrelu_rate': ('float', [0.1, 0.9]),
    'exponential_decay_step': ('cat', [1, 5, 10]),
    
}

def optuna_hpo_and_best_model_evaluation(data, args, result_train_file, specific_params, n_trials=100):
    def objective(trial):
        def get_specific_params(specific_params):
            ans = dict()
            for k, (suggest_type, suggest_param) in specific_params.items():
                if suggest_type == "cat":
                    ans[k] = trial.suggest_categorical(k, suggest_param)
                elif suggest_type == "int":
                    ans[k] = trial.suggest_int(name=k, step=1, low=suggest_param[0], high=suggest_param[1])
                elif suggest_type == "float":
                    print(k, suggest_param)
                    ans[k] = trial.suggest_float(k, *suggest_param)
            return ans

        trial_specific_params = get_specific_params(specific_params)
        
        args.__dict__.update(**trial_specific_params)

        metrics = train(data, args, result_train_file)
        return metrics['wmape']

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    

    return best_params


if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            if args.cross_val is True:
                best_params = optuna_hpo_and_best_model_evaluation(data, args, result_train_file, specific_params)
                print('Best params:', best_params)
                args.__dict__.update(**best_params)
            wmape = []
            rmse = []
            best_score = np.inf
            for _ in range(args.n_retrains):
                m = train(data, args, result_train_file, cross_val=False, best_score=best_score)
                wmape.append(m['wmape'])
                rmse.append(m['rmse'])
                if m['wmape'] < best_score:
                    best_score = m['wmape']
            wmape = np.array(wmape)
            rmse = np.array(rmse)
            
            after_train = datetime.now().timestamp()
            print(f'Score: WMAPE {100 * wmape.mean():.3f} ± {wmape.std():.3%}; RMSE {rmse.mean():.3f} ± {rmse.std():.3f}')
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        test_data = data[-(args.window_size + 30):]
        before_evaluation = datetime.now().timestamp()
        forecast = test(test_data, args, result_train_file)
        forecast = de_normalize(forecast[0])
        df = pd.DataFrame(forecast)
        df.to_csv('forecast/preds.csv')
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')