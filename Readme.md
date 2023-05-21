                                                                  Хакатон(кейс VK)
Здесь вы можете найти решение задачи команды VARCUBIKS для Хакатона по кейсу вк.
Решение задачи по прогнозированию значения переменной x на основе данных train.csv и test.csv с помощью модели XGBoost.
________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/bc83c545-1d4e-4bcc-89af-1c940280f485)
________________________________________________________________________________________________________________________________________


Использование библиотек XGBoost, pandas, numpy. 

________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/a521bcc7-1508-4ccc-a49b-212a28b59782)
________________________________________________________________________________________________________________________________________


Как мы решали задачу:

Преобразование переменной x в логарифм, обработка числовых признаков, удаление ненужных признаков, обучение модели XGBoost с определенными параметрами и сохранение результатов в файл submission.csv. 

Уникальность решения заключается в использовании XGBoost для прогнозирования значений переменной x и преобразовании переменной x в логарифм для улучшения качества модели.

________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/93ebccbd-5c69-43f8-878a-2ec34dcccb43)
________________________________________________________________________________________________________________________________________


В репозитории находятся два решения, которые на наш взгляд обсалютно верны, но проверить их нашей команде не удалось, так как наше оборудование не позволило нам этого сделать. Ниже представлен скрин ошибки, которая возникла у нас при решение задачи:

________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/0866e3fa-f51e-4dfe-8ef2-78b05d179eb7)
________________________________________________________________________________________________________________________________________

Оба решения затратны по памяти и если бы было больше времени, то мы бы сумели оптимизировать его так, чтобы возможно было получить файл csv с верным ответом.


Решения задачи являются индивидуальным, каждый из участников команды участвовал в создание. В первом файле под названием haks представлено самое первое и совместное решение от нашей команды, а второе решение в файле gg уже собрано из 4 разных кодов, которые составляли участники команды.



 ________________________________________________________________________________________________________________________________________
 
 # Зачем наш код нужен?
                                                         
 ________________________________________________________________________________________________________________________________________
 
Наш код предназначен для обучения модели машинного обучения на данных из файлов train.csv, test.csv и attr.csv, а также для создания прогнозов на тестовых данных и сохранения их в файл submission.csv. 
Сначала происходит импорт необходимых библиотек: pandas, numpy и XGBRegressor из библиотеки xgboost. 

 ________________________________________________________________________________________________________________________________________
 
Затем загружаются данные из файлов для анализа. В train_df и test_df сохраняются данные из файлов train.csv и test.csv соответственно, а в attr_df сохраняются данные из файла attr.csv. 

________________________________________________________________________________________________________________________________________
 
Далее происходит объединение данных из train_df и attr_df по столбцам ego_id и u, а также данных из test_df и attr_df по тем же столбцам.

________________________________________________________________________________________________________________________________________

Затем пропущенные значения в столбцах city_id, sex, school и university заполняются значением -1, а затем эти столбцы преобразуются в тип int8. 
Далее целевая переменная x1 преобразуется логарифмически. 

________________________________________________________________________________________________________________________________________

Затем определяются признаки (X_train) и целевая переменная (y_train) на основе данных из train_df, а также признаки на основе данных из test_df (X_test). 

________________________________________________________________________________________________________________________________________

После этого модель XGBRegressor обучается на данных из X_train и y_train с использованием параметров max_depth=4, objective='reg:squarederror' и n_estimators=1000. 
Затем модель используется для создания прогнозов на данных из X_test, которые обратно преобразуются из логарифмической формы в исходную форму. 

________________________________________________________________________________________________________________________________________

Наконец, создаются данные для файла submission.csv на основе данных из test_df и сохраняются в файл submission.csv.

________________________________________________________________________________________________________________________________________

# Финальное решение:

```Python
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
```


________________________________________________________________________________________________________________________________________

# Немного о нашем первом решение- Для решения задачи мы использовали  метод градиентного бустинга, так как он показывает хорошие результаты на задачах регрессии. Вот такие были шаги:


________________________________________________________________________________________________________________________________________


## Шаг 1: Загрузка данных

Начнем с загрузки данных из файлов. Для этого мы будем использовать библиотеку pandas.

## Фрагмент кода:
```Python
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(train_df.head()) #первые строки(для проверки)
print(test_df.head())#вторые строки(для проверки)
```

________________________________________________________________________________________________________________________________________


## Шаг 2: Предобработка данных

Перед тем, как начать обучение модели, необходимо провести предобработку данных. В  задаче  мы использовали следующие признаки:

1. Момент появления связи между пользователями.
2. Возраст пользователей.
3. Идентификатор города.
4. Пол пользователя.
5. Идентификаторы школ и университетов, указанных в профиле.

Для начала посмотрим на распределение целевой переменной в обучающей выборке.

### Фрагмент кода:
```Python
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
```

Мы видим, что признаки age, city, school и university имеют длинные хвосты, поэтому мы можем их прологарифмировать. Признак t имеет отрицательные значения, поэтому мы можем добавить к нему константу 1 и прологарифмировать.

### Фрагмент кода:
```Python
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
```


________________________________________________________________________________________________________________________________________

## Шаг 3: Обучение модели

Теперь мы готовы обучить модель. Для этого мы будем использовать библиотеку XGBoost.

### Фрагмент кода:
```Python
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
```

________________________________________________________________________________________________________________________________________













________________________________________________________________________________________________________________________________________

                                              Связь с нами и участники команды:
                                              
________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/a9e77979-1b96-4b97-9c51-2d7056d2a1a3)
________________________________________________________________________________________________________________________________________

                                                        Наши работяги:
 ________________________________________________________________________________________________________________________________________
 ________________________________________________________________________________________________________________________________________
                                                       А кто это у нас?

________________________________________________________________________________________________________________________________________
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/a46dc3b8-aeec-4ef1-a044-648804e67251)
________________________________________________________________________________________________________________________________________

                                                       Опа, а еще кто?
________________________________________________________________________________________________________________________________________ 
![image](https://github.com/LAITxNIKEYs/hsksvk/assets/104034823/50d4259f-cd05-40e8-b0f2-560d21b56983)
________________________________________________________________________________________________________________________________________


