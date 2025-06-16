# Проект: Определение токсичности комментариев

## Набор данных: 
База данных с комментариями (toxic_comments):
* text — текст комментария
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
