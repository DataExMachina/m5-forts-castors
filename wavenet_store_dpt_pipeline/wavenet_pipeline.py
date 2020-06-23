
import pandas as pd
import numpy as np
import glob
import fire

import gc
import itertools
import functools
import tqdm

import datetime

from gluonts.dataset.common import ListDataset
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer
import mxnet
mxnet.random.seed(3105)

def gluonts_pred_to_m5_median(forecast):
    res = forecast.mean_ts.reset_index()\
                  .rename(columns={'index': 'pred', 0: 'date'})\
                  .set_index('date').reset_index(drop=True)
    res['pred'] = forecast.median
    return res.T

def generate_gp_gluonts_predict(gp_df):
    
    # set date index
    gp_df = gp_df.sort_index()
    
    # create traing and test
    training_list = list()
    valid_list = list()
    
    training_list.append(
        {"start": gp_df.index[0],
         "target": gp_df.iloc[:-28, 0]}
        )
    
    valid_list.append(
        {"start": gp_df.index[0],
         "target": gp_df.iloc[:, 0]}
    )       

    # dump train / test into ListDataset
    training_data = ListDataset(
        training_list,
        freq = "D",
    )

    valid_data = ListDataset(
        valid_list,
        freq = "D",
    )
    
    estimator = WaveNetEstimator(freq="D", prediction_length=28,
                                 act_type='relu',
                                 num_bins=512,
                                 trainer=Trainer(epochs=15))
    predictor = estimator.train(training_data=training_data)
    return gluonts_pred_to_m5_median(list(predictor.predict(valid_data))[0]).T

def pipeline(horizon="validation"):

    # load data
    data = {}
    for path in glob.glob('./data/raw/*.csv'):
        name = path.split('/')[-1].replace('.csv', '')
        if name=="calendar":
            data[name] = pd.read_csv(path, parse_dates=['date'])
            data[name] = data[name].sort_values('date')
            data[name]['event_type_1'].fillna('nothing', inplace=True)
            data[name]['event_type_2'].fillna('nothing', inplace=True)
        else:
            data[name] = pd.read_csv(path)
    
    # prepare sales
    if horizon=="validation":
        sales = pd.melt(data['sales_train_validation'],
                id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                value_vars=['d_%s' % i for i in range(1, 1914)],
                var_name='d', value_name='target')
    elif horizon=="evaluation":
        sales = pd.melt(data['sales_train_evaluation'],
                id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                value_vars=['d_%s' % i for i in range(1, 1942)],
                var_name='d', value_name='target')
    else:
        raise ValueError('Wrong value for horizon')

    sales = sales.merge(data['calendar'][['date', 'd']], how='left')

    # agg sales
    sales_agg = sales.groupby(['date', 'store_id', 'dept_id'])['target'].sum().reset_index()
    sales_agg['key'] = sales_agg[['store_id', 'dept_id']].apply(lambda x: '_'.join(x), axis=1)

    # generate forecasts
    train_set = sales_agg.pivot_table(index='date', columns='key', values='target')
    submit = list()
    for c in sales_agg['key'].drop_duplicates().tolist():
        print('Start', c)
        submit.append(
            generate_gp_gluonts_predict(train_set[[c]])
        )
        print(submit[-1])
    

    predict = pd.concat(submit, axis=1, ignore_index=True)
    predict.columns = sales_agg['key'].drop_duplicates().tolist()
    predict.to_csv('./data/external/forecast_wavenet_store_dpt_%s.csv' % horizon, index=False)

if __name__ == "__main__":
    fire.Fire(pipeline)