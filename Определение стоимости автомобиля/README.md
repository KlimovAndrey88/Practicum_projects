# Проект: Определение стоимости автомобилей

## Набор данных: 
Технические характеристики, комплектации и цены автомобилей:
* DateCrawled — дата скачивания анкеты из базы
* VehicleType — тип автомобильного кузова
* RegistrationYear — год регистрации автомобиля
* Gearbox — тип коробки передач
* Power — мощность (л. с.)
* Model — модель автомобиля
* Kilometer — пробег (км)
* RegistrationMonth — месяц регистрации автомобиля
* FuelType — тип топлива
* Brand — марка автомобиля
* Repaired — была машина в ремонте или нет
* DateCreated — дата создания анкеты
* NumberOfPictures — количество фотографий автомобиля
* PostalCode — почтовый индекс владельца анкеты (пользователя)
* LastSeen — дата последней активности пользователя
* Price — цена (евро)

## Цель проекта: 
Построение модели для определения стоимости автомобиля по его характеристикам. 

## Основные шаги исследования: 
* Подготовка данных.
* Предобработка данных.
* Исследовательский анализ данных.
* Обучение моделей.
* Анализ моделей и выбор лучшей.
* Общий вывод.

## Используемые библиотеки и модули:
* sklearn (OneHotEncoder, OrdinalEncoder, StandardScaler,  MinMaxScaler, train_test_split, RandomizedSearchCV, KFold, LinearRegression, DecisionTreeRegressor, mean_squared_error, make_scorer, Pipeline, ColumnTransformer, SimpleImputer)
* catboost (CatBoostRegressor)
* lightgbm (LGBMRegressor)
* category_encoders (TargetEncoder)
* math
* pandas
* matplotlib.pyplot 
* numpy 
* seaborn 
* scipy
* missingno
* phik_matrix
* plot_correlation_matrix
* import shap

## Выводы:
* Все обученные модели классов LGBMRegressor и CatBoostRegressor, а также четыре модели класса DecisionTreeRegresso удовлетворяют критерию RMSE < 2500. Самый лучший показатель у CatBoostRegressor.
* Самое лучшее время обучения и прогнозирования у класса DecisionTreeRegresson. Самое большое время у класса LGBMRegressor.
* Для прогнозирования стоимости автомобилей рекомендуется использовать лучшую модель класса CatBoostRegressor с RMSE=1544.37, временем обучения = 37.04 мс и временем прогнозирования = 892 ms мс (на тестовой выборке).
