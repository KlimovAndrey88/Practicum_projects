# Проект: Прогнозирование заказов такси

## Набор данных: 
Данные о количестве заказов такси:
* datetime (индекс) - время заказов
* num_orders — количество заказов

## Цель проекта: 
Построение модели для прогнозирования количества заказов такси на следующий час. 

## Основные шаги исследования: 
* Подготовка данных.
* Исследовательский анализ данных.
* Обучение моделей.
* Тестирование лучшей модели.
* Общий вывод.

## Используемые библиотеки и модули:
* sklearn (StandardScaler,  MinMaxScaler, train_test_split, RandomizedSearchCV, TimeSeriesSplit, LinearRegression, DecisionTreeRegressor, mean_squared_error, root_mean_squared_error, make_scorer, Pipeline, ColumnTransformer, SimpleImputer)
* catboost (CatBoostRegressor)
* lightgbm (LGBMRegressor)
* math
* pandas
* matplotlib.pyplot 
* numpy 
* seaborn 
* scipy
* seasonal_decompose
* shap

## Выводы:
* Самый лучший показатель у CatBoostRegressor. На тестовой выборке данная модель показала метрику RMSE = 43.63.
* Для прогнозирования заказов такси на следующий час рекомендуется использовать модель класса CatBoostRegressor с гиперпараметрами:
  * 'preprocessor__num': MinMaxScaler()
  * 'models__learning_rate': 0.1
  * 'models__l2_leaf_reg': 5
  * 'models__early_stopping_rounds': 40
  * 'models__depth': 3
