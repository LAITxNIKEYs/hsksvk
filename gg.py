import pandas as pd
import numpy as np
from xgboost import XGBRegressor


train_df = pd.read_csv("train.csv", dtype={'city_id': 'int8', 'sex': 'int8', 'school': 'int16', 'university': 'int16'})
test_df = pd.read_csv("test.csv", dtype={'city_id': 'int8', 'sex': 'int8', 'school': 'int16', 'university': 'int16'})

attr_df = pd.read_csv('attr.csv', dtype={'ego_id': 'int32', 'u': 'int32'})
train_df = pd.merge(train_df, attr_df, on=['ego_id','u'], how='left')
test_df = pd.merge(test_df, attr_df, on=['ego_id','u'], how='left')


train_df[['city_id','sex','school','university']] = train_df[['city_id','sex','school','university']].fillna(-1).astype('int8')
test_df[['city_id','sex','school','university']] = test_df[['city_id','sex','school','university']].fillna(-1).astype('int8')

# Преобразуем целевую переменную логарифмически
train_df['x1'] = np.log(train_df['x1'])

# Выделяем признаки и целевую переменную
X_train = train_df[['v','t','x2','x3','age','city_id','school','university']]
y_train = train_df['x1']
X_test = test_df[['v','t','x2','x3','age','city_id','school','university']]

# Обучаем модель
model = XGBRegressor(max_depth=4, objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# Получаем прогнозы и обратно преобразуем целевую переменную
predictions = np.exp(model.predict(X_test))

# Сохраняем прогнозы в файл
submission = test_df[['ego_id', 'u', 'v']].copy()
submission['x1'] = predictions
submission.sort_values(['ego_id', 'u', 'v']).to_csv('submission.csv', index=False)
print("Готово!")