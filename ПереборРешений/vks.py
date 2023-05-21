import pandas as pd
import numpy as np
from xgboost import XGBRegressor

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

attr_df = pd.read_csv('attr.csv')
train_df = pd.merge(train_df, attr_df, on=['ego_id','u'], how='left')
test_df = pd.merge(test_df, attr_df, on=['ego_id','u'], how='left')

train_df[['city_id','sex','school','university']] = train_df[['city_id','sex','school','university']].fillna(-1).astype(np.int32)
test_df[['city_id','sex','school','university']] = test_df[['city_id','sex','school','university']].fillna(-1).astype(np.int32)

train_df['x1'] = np.log(train_df['x1'])

X_train = train_df[['v','t','x2','x3','age','city_id','school','university']]
y_train = train_df['x1']
X_test = test_df[['v','t','x2','x3','age','city_id','school','university']]

model = XGBRegressor(max_depth=4, objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

predictions = np.exp(model.predict(X_test))

submission = test_df[['ego_id', 'u', 'v']].copy()
submission['x1'] = predictions
submission.sort_values(['ego_id', 'u', 'v']).to_csv('submission.csv', index=False)
print("Готово!")
