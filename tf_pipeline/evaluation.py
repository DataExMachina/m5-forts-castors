<<<<<<< HEAD
"""
EVALUATION FILE
gathered on Kaggle

"""
=======
'''
EVALUATION FILE
gathered on Kaggle

'''
>>>>>>> parent of e6adfb9... nouveau pipeline
import numpy as np
import pandas as pd
from typing import Union
from tqdm.notebook import tqdm
from conf import RAW_PATH
import os

import pickle
from pprint import pprint
<<<<<<< HEAD

with open("../data/models/best_params.pkl", "rb") as f:
=======
with open('../data/models/best_params.pkl', 'rb') as f:
>>>>>>> parent of e6adfb9... nouveau pipeline
    data = pickle.load(f)

pprint(data)

# class WRMSSEEvaluator(object):
#
#     def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
#         train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
#         train_target_columns = train_y.columns.tolist()
#         weight_columns = train_y.iloc[:, -28:].columns.tolist()
#
#         train_df['all_id'] = 0  # for lv1 aggregation
#
#         id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
#         valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()
#
#         if not all([c in valid_df.columns for c in id_columns]):
#             valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)
#
#         self.train_df = train_df
#         self.valid_df = valid_df
#         self.calendar = calendar
#         self.prices = prices
#
#         self.weight_columns = weight_columns
#         self.id_columns = id_columns
#         self.valid_target_columns = valid_target_columns
#
#         weight_df = self.get_weight_df()
#
#         self.group_ids = (
#             'all_id',
#             'cat_id',
#             'state_id',
#             'dept_id',
#             'store_id',
#             'item_id',
#             ['state_id', 'cat_id'],
#             ['state_id', 'dept_id'],
#             ['store_id', 'cat_id'],
#             ['store_id', 'dept_id'],
#             ['item_id', 'state_id'],
#             ['item_id', 'store_id']
#         )
#
#         for i, group_id in enumerate(tqdm(self.group_ids)):
#             train_y = train_df.groupby(group_id)[train_target_columns].sum()
#             scale = []
#             for _, row in train_y.iterrows():
#                 series = row.values[np.argmax(row.values != 0):]
#                 scale.append(((series[1:] - series[:-1]) ** 2).mean())
#             setattr(self, f'lv{i + 1}_scale', np.array(scale))
#             setattr(self, f'lv{i + 1}_train_df', train_y)
#             setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())
#
#             lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
#             setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())
#
#     def get_weight_df(self) -> pd.DataFrame:
#         day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
#         weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
#         weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
#         weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)
#
#         weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
#         weight_df['value'] = weight_df['value'] * weight_df['sell_price']
#         weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
#         weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
#         weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
#         return weight_df
#
#     def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
#         valid_y = getattr(self, f'lv{lv}_valid_df')
#         score = ((valid_y - valid_preds) ** 2).mean(axis=1)
#         scale = getattr(self, f'lv{lv}_scale')
#         return (score / scale).map(np.sqrt)
#
#     def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
#         assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape
#
#         if isinstance(valid_preds, np.ndarray):
#             valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)
#
#         valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)
#
#         group_ids = []
#         all_scores = []
#         for i, group_id in enumerate(self.group_ids):
#             lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
#             weight = getattr(self, f'lv{i + 1}_weight')
#             lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
#             group_ids.append(group_id)
#             all_scores.append(lv_scores.sum())
#
#         return group_ids, all_scores
#
# ## reading data
# df_train_full = pd.read_csv(RAW_PATH+"sales_train_evaluation.csv")
#
# df_calendar = pd.read_csv(RAW_PATH+"calendar.csv")
# df_prices = pd.read_csv(RAW_PATH+"sell_prices.csv")
# df_sample_submission = pd.read_csv(RAW_PATH+"sample_submission.csv")
# df_sample_submission["order"] = range(df_sample_submission.shape[0])
#
# df_train = df_train_full.iloc[:, :-28]
# df_valid = df_train_full.iloc[:, -28:]
#
# evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)
# path_predictions = os.path.join(SUBMIT_PATH, "tf_estim_%s.csv" % horizon)
<<<<<<< HEAD
# df = pd.read_csv(path_predictions)
=======
# df = pd.read_csv(path_predictions)
>>>>>>> parent of e6adfb9... nouveau pipeline
