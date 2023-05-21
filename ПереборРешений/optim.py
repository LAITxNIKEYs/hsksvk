import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Уменьшаем использование памяти при чтении данных
train_df = pd.read_csv("train.csv", dtype={'city_id': 'int8', 'sex': 'int8', 'school': 'int16', 'university': 'int16'})
test_df = pd.read_csv("test.csv", dtype={'city_id': 'int8', 'sex': 'int8', 'school': 'int16', 'university': 'int16'})

attr_df = pd.read_csv('attr.csv', dtype={'ego_id': 'int32', 'u': 'int32'})
train_df = pd.merge(train_df, attr_df, on=['ego_id','u'], how='left')
test_df = pd.merge(test_df, attr_df, on=['ego_id','u'], how='left')

# Заполняем пропущенные значения и изменяем тип данных
train_df[['city_id','sex','school','university']] = train_df[['city_id','sex','school','university']].fillna(-1).astype('int8')
test_df[['city_id','sex','school','university']] = test_df[['city_id','sex','school','university']].fillna(-1).astype('int8')

# Преобразуем целевую переменную логарифмически
train_df['x1'] = np.log(train_df['x1'])

# Выделяем признаки и целевую переменную
X_train = train_df[['v','t','x2','x3','age','city_id','school','university']].astype('float32')
y_train = train_df['x1'].astype('float32')
X_test = test_df[['v','t','x2','x3','age','city_id','school','university']].astype('float32')

# Обучаем модель
model = XGBRegressor(max_depth=4, objective='reg:squarederror', n_estimators=1000)

# Обучаем модель по частям (chunking)
chunk_size = 100000
for i in range(0, len(X_train), chunk_size):
    X_chunk = X_train[i:i+chunk_size]
    y_chunk = y_train[i:i+chunk_size]
    model.fit(X_chunk, y_chunk)

# Получаем прогнозы и обратно преобразуем целевую переменную
predictions = []
chunk_size = 100000
for i in range(0, len(X_test), chunk_size):
    X_chunk = X_test[i:i+chunk_size]
    predictions_chunk = np.exp(model.predict(X_chunk))
    predictions.append(predictions_chunk)
predictions = np.concatenate(predictions)

# Сохраняем прогнозы в файл
submission = test_df[['ego_id', 'u', 'v']].copy()
submission['x1'] = predictions
submission.sort_values(['ego_id', 'u', 'v']).to_csv('submission.csv', index=False)
print("Готово!")