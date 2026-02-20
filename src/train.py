from src.preprocessor import load_data, main_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import os
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.logger import get_logger
from features import num_features, cat_binary_features, cat_multiclass_features

# инициируем логгирование
logger = get_logger('train')
logger.info('Training started')

df = load_data()
df = main_preprocessor(df)

# Отделяем целевую переменную (y) от датасета (X)
y = df['Отток клиента']
X = df.drop(['Отток клиента', 'Идентификатор'], axis=1)

# Train/test split. Тест 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# сверка размерности после сплита
logger.info(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

# создаем StandardScaler для числовых признаков
numeric_transformer = StandardScaler()

# создаем OneHotEncoder для бинарных признаков
binary_transformer = OneHotEncoder(
    drop='if_binary',
    handle_unknown='ignore',
    sparse_output=False
    )

# создаем OneHotEncoder для многозначных признаков
multiclass_transformer = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

# создаем препроцессор для трех типов данных
local_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat_bin', binary_transformer, cat_binary_features),
        ('cat_multi', multiclass_transformer, cat_multiclass_features)
    ]
)

# Pipeline модель LogisticRegression
pipeline_pipe = Pipeline(steps=[
    ('preprocessor', local_preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

pipeline_pipe.fit(X_train, y_train)

y_pred = pipeline_pipe.predict(X_test)
y_probs = pipeline_pipe.predict_proba(X_test)[:, 1]

# метрики
metrics = {
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_probs)),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

logger.info(f'Metrics: \n%s', json.dumps(metrics, indent=4))

# сохранение метрик
os.makedirs('artifacts', exist_ok=True)

joblib.dump(pipeline_pipe, 'artifacts/model.joblib')

with open('artefacts/matrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info('Model saved to artefacts')