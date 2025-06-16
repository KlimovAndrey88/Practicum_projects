# Проект: Определение возраста покупателей

## Набор данных: 
Набор фотографий людей с указанием возраста (faces).

## Цель проекта: 
Построение модели для определения приблизительного возраста людей по фотографии. 

## Основные шаги исследования: 
* Исследовательский анализ данных.
* Подготовка данных к обучению
* Обучение нейронной сети и расчет ее качества.

## Используемые библиотеки и модули:
* pandas
* matplotlib.pyplot 
* numpy 
* PIL (Image)
* tensorflow.keras.layers (GlobalAveragePooling2D, Dense)
* tensorflow.keras.models (Sequential)
* tensorflow.keras.applications.resnet (ResNet50)
* tensorflow.keras.preprocessing.image (ImageDataGenerator)
* tensorflow.keras.optimizers (Adam)

## Выводы:
Обученная модель имеет высокую точность предсказания возраста покупателей. Модель имеет метрику качества равную 5,95, это означает, что погрешность в определении возраста составляет примерно 6 лет, что является хорошим показателем.


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


# Проект: Определение токсичности комментариев

## Набор данных: 
База данных с комментариями (toxic_comments):
* text - текст комментария
* toxic — метка токсичности комментария (целевой признак)

## Цель проекта: 
Построение модели для классификации комментариев на позитивные и негативные.

## Основные шаги исследования: 
Подготовка данных.
* Обучение и выбор лучшей модели.
* Общий вывод.

## Используемые библиотеки и модули:
* sklearn (train_test_split, RandomizedSearchCV, TfidfVectorizer, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, f1_score, Pipeline, LabelEncoder)
* pandas
* matplotlib.pyplot 
* numpy 
* seaborn 
* scipy (uniform, randint)
* nltk.corpus (stopwords)
* nltk.stem (WordNetLemmatizer)
* spacy

## Выводы: 
Для классификации комментариев на позитивные и негативные рекомендуется использовать модель класса LogisticRegression с гиперпараметрами {'clf__C': 11, 'clf__penalty': 'l2', 'tfidf__max_df': 0.9852142919229747, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2)}.


# Проект: Определение факторов влияющих на стоимость квартир

## Набор данных: 
Архив объявлений сервиса Яндекс Недвижимость о продаже квартир в Санкт-Петербурге и соседних населенных пунктах за несколько лет, содержащий следующую информацию:
* airports_nearest — расстояние до ближайшего аэропорта в метрах (м)
* balcony — число балконов
* ceiling_height — высота потолков (м)
* cityCenters_nearest — расстояние до центра города (м)
* days_exposition — сколько дней было размещено объявление (от публикации до снятия)
* first_day_exposition — дата публикации
* floor — этаж
* floors_total — всего этажей в доме
* is_apartment — апартаменты (булев тип)
* kitchen_area — площадь кухни в квадратных метрах (м²)
* last_price — цена на момент снятия с публикации
* living_area — жилая площадь в квадратных метрах (м²)
* locality_name — название населённого пункта
* open_plan — свободная планировка (булев тип)
* parks_around3000 — число парков в радиусе 3 км
* parks_nearest — расстояние до ближайшего парка (м)
* ponds_around3000 — число водоёмов в радиусе 3 км
* ponds_nearest — расстояние до ближайшего водоёма (м)
* rooms — число комнат
* studio — квартира-студия (булев тип)
* total_area — общая площадь квартиры в квадратных метрах (м²)
* total_images — число фотографий квартиры в объявлении

## Цель проекта: 
Определить параметры, влияющие на цену объектов недвижимости, для дальнейшего их использования в автоматизированной системе отслеживания аномалий и мошеннической деятельности. 

## Основные шаги исследования: 
* Изучение общей информации о датафрейме и построение гистограмм по всем параметрам.
* Предобработка данных.
* Исследовательский анализ данных.
* Общий вывод.

## Используемые библиотеки и модули:
* pandas
* matplotlib.pyplot
* numpy

## Выводы:
* Изучена скорость продажи квартир.
* Определены факторы, которые больше всего влияют на общую (полную) стоимость объекта.
* Рассчитана средняя цена одного квадратного метра в 10 населённых пунктах с наибольшим числом объявлений.
* Рассчитана средняя цена одного квадратного метра в зависимости от удаленности от центра.


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


# Проект: Прогнозирование заказов такси

Набор данных (нескольких файлов, полученных из разных источников): 
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
