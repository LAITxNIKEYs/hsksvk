import os
import optuna
import requests
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool, MetricVisualizer, CatBoostRegressor
from copy import deepcopy


attr = pd.read_csv('attr.csv')
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission = pd.read_csv('submission.csv')


train_df = pd.merge(train_df, attr, on=['ego_id','u'], how='left')
train_df[['city_id','sex','school','university']] = train_df[['city_id','sex','school','university']].fillna(-1).astype(np.int32)
train_df = train_df.sample(100_000)

test_df = pd.merge(test_df, attr, on=['ego_id','u'], how='left')
test_df[['city_id','sex','school','university']] = test_df[['city_id','sex','school','university']].fillna(-1).astype(np.int32)

sub = pd.merge(test_df, submission[['ego_id','u','v']], on=['ego_id','u','v']).drop_duplicates(subset=['ego_id','u','v']).fillna(-1)
sub[['v','city_id','school','university']] = sub[['v','city_id','school','university']].astype(np.int32)


X_train = train_df[['v','t','x2','x3','age','city_id','school','university']]
y_train = train_df['x1']

X_test = test_df.dropna(subset=['x1'])[['v','t','x2','x3','age','city_id','school','university']]
y_test = test_df.dropna(subset=['x1'])['x1']

train = Pool(data=X_train, label=y_train, cat_features=['v','city_id','school','university'])
test = Pool(data=X_test, label=y_test, cat_features=['v','city_id','school','university'])


model = CatBoostRegressor(depth=4, objective='Poisson', task_type="GPU")
model.fit(train, plot=True, eval_set=test, verbose=500)


sub['x1'] = model.predict(sub[['v','t','x2','x3','age','city_id','school','university']]).astype(float)
sub.sort_values(['ego_id','u','v']).to_csv('sub.csv', index=False, columns=['ego_id','u','v','x1'])
print("Все хорошо, проверяй на alcups")