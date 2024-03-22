from prophet import Prophet
import pandas as pd

def prophet_run(train_prophet, valid_prophet):
    p = list()
    for atm in train_prophet['atm'].unique():
        print('atm:', atm)
        train_ = train_prophet.loc[train_prophet['atm'] == atm]
        valid_ = valid_prophet.loc[valid_prophet['atm'] == atm]
        valid_['ds'] = valid_['ds'].astype('datetime64[ns]')

        m = Prophet()
        m.fit(train_)

        future = m.make_future_dataframe(periods=int(valid_prophet.shape[0] / 88), include_history=False)
        future = future.merge(valid_[['ds']], on='ds', how='left')
        forecast = m.predict(future)
        forecast['atm'] = atm
        p.append(forecast[['ds', 'yhat', 'atm','yhat_lower','yhat_upper','trend']])

    p = pd.concat(p, ignore_index=True)
    p['ds'] = p['ds'].astype('datetime64[ns]')
    valid_prophet['ds'] = valid_prophet['ds'].astype('datetime64[ns]')
    p = p.merge(valid_prophet, on=['ds', 'atm'], how='left')
    p = p.fillna(value=0)
    return p