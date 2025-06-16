# Проект: Прогнозирование оттока клиентов

## Набор данных (нескольких файлов, полученных из разных источников): 
* contract_new.csv — информация о договоре;
* personal_new.csv — персональные данные клиента;
* internet_new.csv — информация об интернет-услугах;
* phone_new.csv — информация об услугах телефонии.

Файл contract_new.csv
* customerID — идентификатор абонента;
* BeginDate — дата начала действия договора;
* EndDate — дата окончания действия договора;
* Type — тип оплаты: раз в год-два или ежемесячно;
* PaperlessBilling — электронный расчётный лист;
* PaymentMethod — тип платежа;
* MonthlyCharges — расходы за месяц;
* TotalCharges — общие расходы абонента.

Файл personal_new.csv
* customerID — идентификатор пользователя;
* gender — пол;
* SeniorCitizen — является ли абонент пенсионером;
* Partner — есть ли у абонента супруг или супруга;
* Dependents — есть ли у абонента дети.

Файл internet_new.csv
* customerID — идентификатор пользователя;
* InternetService — тип подключения;
* OnlineSecurity — блокировка опасных сайтов;
* OnlineBackup — облачное хранилище файлов для резервного копирования данных;
* DeviceProtection — антивирус;
* TechSupport — выделенная линия технической поддержки;
* StreamingTV — стриминговое телевидение;
* StreamingMovies — каталог фильмов.

Файл phone_new.csv
* customerID — идентификатор пользователя;
* MultipleLines — подключение телефона к нескольким линиям одновременно.

## Цель проекта: 
Построение модели для прогнозирования оттока клиентов оператора связи.

## Основные шаги исследования: 
* Загрузка данных и их первичный осмотр.
* Исследовательский анализ и предобработка данных.
* Объединение данных.
* Исследовательский анализ и предобработка данных объединенного датафрейма.
* Подготовка данных.
* Обучение моделей машинного обучения.
* Выбор лучшей модели.
* Общий вывод и рекомендации заказчику.

## Используемые библиотеки и модули:
* sklearn (train_test_split, RandomizedSearchCV, OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression, make_pipeline, Pipeline, ColumnTransformer, SimpleImputer, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, f1_score)
* lightgbm (LGBMClassifier)
* catboost (CatBoostClassifier)
* missingno
* pandas
* matplotlib.pyplot 
* numpy 
* seaborn 
* scipy (uniform, randint)
* phik (phik_matrix, plot_correlation_matrix)
* shap

 ##  Выводы:
Рекомендации для бизнеса: Для прогнозирования оттока клиентов с целью проведения акций и выдачи промокодов рекомендуется использовать модель класса CatBoostClassifier. Важнейшими признаками являются:
* срок действия договора (num__contract_duration)
* общие расходы (num__total_charges)
* количество подключенных услуг (num__num_services)
