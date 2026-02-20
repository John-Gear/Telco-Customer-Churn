from src.preprocessor import load_data, main_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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

# формируем список числовых признаков
num_features = [
    'Стаж клиента',
    'Ежемесячные платежи',
    'Совокупные платежи за всё время',
    'Пожилой клиент'
]

# формируем список бинарных признаков
cat_binary_features = [
    'пол',
    'Наличие супруга(и)',
    'Иждивенцы',
    'Наличие телефонной связи',
    'Электронный счёт'
]

# формируем список многозначных признаков
cat_multiclass_features = [
    'Несколько телефонных линий',
    'Тип интернет-подключения',
    'Онлайн-защита',
    'Онлайн-резервное копирование',
    'Защита устройств',
    'Техническая поддержка',
    'Стриминговое телевидение',
    'Стриминг фильмов',
    'Тип контракта',
    'Способ оплаты'
]

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
