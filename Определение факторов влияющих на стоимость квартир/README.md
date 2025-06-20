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
