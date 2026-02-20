import joblib

MODEL_PATH = 'artifacts/model.joblib'
THRESHOLD = 0.2869508152860422 # т.к. у нас нет единого env, один раз устанавливаем тут порог

def load_model():
    return joblib.load(MODEL_PATH)

# df_features: получаем по апи/бд новых клиентов, превращаем их в pandas.DataFrame
# предсказываем вероятности
def predict_proba(df_features):
    model = load_model,
    probs = model.predict_proba(df_features)[:, 1]
    return probs

# предсказываем класс 0/1 (при пороге threshold=0.2869508152860422)
def predict(df_features, threshold=THRESHOLD):
    probs = predict_proba(df_features)
    preds = (probs >= threshold).astype(int)
    return preds