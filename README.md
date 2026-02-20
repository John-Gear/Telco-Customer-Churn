# Telco Customer Churn Prediction (ML + API + Docker)

Проект решает задачу **бинарной классификации оттока клиентов** (customer churn) на табличных данных Telco.  
Фокус проекта — **инженерный ML-пайплайн**: данные в SQLite, обучение через `scikit-learn Pipeline`, сохранение артефактов, inference через Flask API и запуск в Docker.

---

## Ключевые идеи проекта

- **Источник данных — SQLite**
- **Модель сохранена как артефакт** `artefacts/model.joblib` (внутри весь Pipeline: scaler + encoders + logistic regression).
- **Inference отделён от обучения**: API не обучает модель, API только делает предсказания по готовому артефакту.
- **Контракт признаков** вынесен в `features.py`, чтобы одинаково использовать списки колонок в `train.py` и `app.py`.

---

## Структура проекта

```text
telco_churn/
│
├── notebooks/
│       └── Telco notebook.ipynb          # исследовательский ноутбук
│
├── data/
│   ├── csv/
│   │   └── Telco-Customer-Churn.csv
│   └── sql/
│       └── Telco-Customer-Churn.db       # SQLite база данных
│
├── artefacts/
│   ├── model.joblib                      # сохранённый sklearn inference артефакт
│   └── metrics.json                      # метрики
│
├── src/
│   ├── db.py                             # загрузка данных из SQLite
│   ├── preprocessor.py                   # чистка/приведение типов (без split и без fit моделей)
│   ├── features.py                       # списки колонок: num/cat_binary/cat_multiclass
│   ├── train.py                          # обучение + сохранение model.joblib и metrics.json
│   ├── predict.py                        # порог THRESHOLD и вспомогательная логика inference
│   └── logger.py                         # логгер
│
├── app.py                                # Flask API (/health, /predict)
├── requirements.txt
├── entrypoint.sh                         # логика запуска обучения модели (без обученной модели, inference артефактов) контейнер не стартует
└── Dockerfile
```

---

## Данные

В проекте используется датасет Telco Customer Churn, загруженный в SQLite:

- файл: data/sql/Telco-Customer-Churn.db
- таблица: telco_customers

---

## ML пайплайн

### **Обучение (train)**
1. Загрузка данных из SQLite
2. Базовая чистка (/preprocessor.py)
3. Разделение на X/y, train/test split
3. Сборка ColumnTransformer, StandardScaler для числовых, OneHotEncoder для бинарных категориальных, OneHotEncoder для мультиклассовых категориальных
4. Обучение модели (Logistic Regression) внутри Pipeline
5. Сохранение артефактов: artefacts/model.joblib, artefacts/metrics.json

### **Inference (predict/API)**
- В artefacts/model.joblib сохранён весь Pipeline, поэтому на вход API можно подавать признаки (X) как в обучении.
- В API есть валидация признаков (EXPECTED_COLS из src/features.py) чтобы модель не упала.

---

## Flask API Endpoints
- GET /health — проверка, что сервис жив
- POST /predict — предсказания
- API принимает один объект (клиент) или список объектов (клиенты)
- Валидация входяших данных (проверка колонок, удаление лишних)

---

## Docker

1. Сборка образа
```bash
docker build -t telco-api .
```

2. Запуск контейнера
```bash
docker run -p 5000:5000 telco-api
# или любой свободный порт
```

- при запуске Dockerfile - entrypoint.sh сначала обучит модель, получить артефакты pipeline (model.joblib) и только потом стартует flask приложение

---

## Результаты

- Результаты обучения модели зафиксированы в artefacts/metrics.json.
- Logistic Regression используется как базовая и интерпретируемая модель по ROC-AUC в данном проект (про выбор модели, тюнинг, регуляризацию, etc. почитать в Telco notebook.ipynb)

---

## Воспроизводимый ноутбук на Kaggle

https://www.kaggle.com/code/johngearonline/teclo-customer-churn-ml-pipeline-johngear