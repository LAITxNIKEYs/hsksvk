#_________________________________________________Данные_________________________________________________
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(train_df.head()) #первые строки
print(test_df.head())#вторые строки
#_________________________________________________Предобработка и обработка_________________________________________________
import seaborn as sns
import numpy as np


sns.distplot(train_df['x'])
train_df['log_x'] = np.log(train_df['x']) #преобразование в логарифм
sns.distplot(train_df['log_x'])
#проверка числовых признаков
sns.distplot(train_df['age'])
sns.distplot(train_df['city'])
sns.distplot(train_df['school'])
sns.distplot(train_df['university'])
sns.distplot(train_df['t']) #значение отрицательно


train_df['log_age'] = np.log(train_df['age'])
train_df['log_city'] = np.log(train_df['city'])
train_df['log_school'] = np.log(train_df['school'])
train_df['log_university'] = np.log(train_df['university'])
train_df['log_t'] = np.log(train_df['t'] + 1)
test_df['log_age'] = np.log(test_df['age'])
test_df['log_city'] = np.log(test_df['city'])
test_df['log_school'] = np.log(test_df['school'])
test_df['log_university'] = np.log(test_df['university'])
test_df['log_t'] = np.log(test_df['t'] + 1) #преобразуем с константой 1
#удаление нужных признаков
train_df = train_df.drop(['x', 'age', 'city', 'school', 'university', 't'], axis=1)
test_df = test_df.drop(['age', 'city', 'school', 'university', 't'], axis=1)
#_________________________________________________Обучение_________________________________________________
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#разделение
X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop('log_x', axis=1), train_df['log_x'], test_size=0.2, random_state=42)
#преобразование данных в формат DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(test_df)
#определение параметров
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}



#Обучение модели
model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')], early_stopping_rounds=50, verbose_eval=50)
y_pred = model.predict(dtest) #предсказание на тесте
y_pred = np.exp(y_pred) #обратный лагорифм 
#_________________________________________________Сохранение результатов_________________________________________________
submission = pd.DataFrame({'Id': test_df.index, 'x': y_pred})
submission.to_csv("submission.csv", index=False)
