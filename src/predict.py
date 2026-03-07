import joblib
import pandas as pd
from src.data_cleaner import main_cleaner
from src.features import num_features, cat_binary_features, cat_multiclass_features

MODEL_PATH = 'artifacts/model.joblib'
THRESHOLD = 0.2869508152860422 # т.к. у нас нет единого env, один раз устанавливаем тут порог

EXPECTED_COLS = num_features + cat_binary_features + cat_multiclass_features

def load_model():
    return joblib.load(MODEL_PATH)

# прогоняем новые данные через data_cleaner и удаляем customerID
def prepare_features(df_features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_features, pd.DataFrame):
        df_features = pd.DataFrame(df_features)

    df = df_features.copy()

    # очистка через data_cleaner
    df = main_cleaner(df)

    # удаляем customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Не хватает колонок для инференса: {missing}')

    return df[EXPECTED_COLS]

# df_features: получаем по api/бд новых клиентов, превращаем их в pandas.DataFrame
# предсказываем вероятности
def predict_proba(df_features):
    model = load_model()
    df_prepared = prepare_features(df_features)
    probs = model.predict_proba(df_prepared)[:, 1]
    return probs

# предсказываем класс 0/1 (при пороге threshold=0.2869508152860422)
def predict(df_features, threshold=THRESHOLD):
    probs = predict_proba(df_features)
    preds = (probs >= threshold).astype(int)
    return preds