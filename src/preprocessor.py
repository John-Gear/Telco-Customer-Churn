from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import num_features, cat_binary_features, cat_multiclass_features

def main_preprocessor():
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

    return local_preprocessor