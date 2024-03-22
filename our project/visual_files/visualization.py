import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize(test_pred, train, p):

    atm_machines = np.array([  0, 108, 105,   7,  86,  18,  19,  64,  38,  90,  83,  24,  78,
            69,  33, 102,  36,  35,  67, 109,  46,  56,   3,  32,   4,  70,
        107,  74,  66,  99, 103,  14,  91,  17,  50,  52,   9,  82,  25,
        104,   8,  77,  22, 106,   6,  42,  60,   5,  21,  27,  75,  30,
            58,  10,  48,  59,  84,  55,  54,  34,  43,  62,  40,  65,  41,
            61,  31,  71,  44,  47,  45,  68,  79,  57,  76,  26,  23,  37,
            20,  98,  12,  39, 100,  63,  16])

    a3t_gcn = pd.DataFrame(np.array(test_pred.cpu().detach().numpy()))
    a3t_gcn = a3t_gcn.set_index(pd.date_range(start='12/02/2017', end='12/31/2017'))
    a3t_gcn.columns = atm_machines
    #a3t_gcn.to_csv(f'{model_name}_yhat.csv')
    #tgcn_predict = pd.read_csv('a3t_gcn_yhat.csv', index_col=0)
    tgcn_predict = pd.melt(a3t_gcn.reset_index(), id_vars='index')
    tgcn_predict = tgcn_predict.rename(columns={'index':'ds','variable':'atm','value':'y_hat'})
    make_plot(tgcn_predict, pd.DataFrame(train), p)

def make_plot(tgcn_predict, train, p):
    fig, ax = plt.subplots(3,2, figsize=(18, 5), dpi=300)
    plt.rcParams['font.size'] = '22'
    for ax_,atm in zip(ax,tgcn_predict['atm'].unique()[10:16]):
        tgcn_predict['atm'] = tgcn_predict['atm'].astype('int')
        tgcn_predict['ds'] = tgcn_predict['ds'].astype('datetime64[ns]')

        tgcn_predict_ = tgcn_predict.loc[tgcn_predict['atm'] == atm]
        t_ = train.loc[train['atm'] == atm]
        p_ = p.loc[p['atm'] == atm]

        t_['ds'] = t_['ds'].astype('datetime64[ns]')

        t_.plot(x='ds', y='y', ax=ax_, label='historic')
        p_.plot(x='ds', y='y', ax=ax_, label='future')
        tgcn_predict_.plot(x='ds', y='y_hat', ax=ax_, label='predict')

        ax_.set_title(atm)
        ax_.legend(fontsize='20', loc='upper left')

        # ax[ax_].set_xlabel('Date')
        ax_.set_ylabel('transactions')
        print(np.abs(p_['y'] - tgcn_predict_['y_hat']).sum() / np.abs(p_['y']).sum())
    fig.tight_layout()
    plt.show()