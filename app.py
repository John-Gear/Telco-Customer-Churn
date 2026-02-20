from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.logger import get_logger
from src.predict import THRESHOLD
from src.features import num_features, cat_binary_features, cat_multiclass_features

logger = get_logger('api')

app = Flask(__name__)

MODEL_PATH = 'artifacts/model.joblib'
ID_COL = 'Идентификатор'
APP_THRESHOLD = THRESHOLD # забираем порог из predict.py т.к. у нас нет единого env. Принято решение хранить порог только там
EXPECTED_COLS = num_features + cat_binary_features + cat_multiclass_features # наши колонки,  чтобы зафиксировать контракт инференс vs train

# один раз загружаем модель при старте сервера
model = joblib.load(MODEL_PATH)
logger.info('API started, model load')

# проверка жив ли сервер
@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

# берем json из тела запроса
@app.post('/predict')
def predict():
    payload = request.get_json()

    if isinstance(payload, dict): # можем принять либо 1 клиента, либо список клиентов
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return jsonify({'error': 'JSON должен быть object или list'}), 400
    
    df = pd.DataFrame(rows)

    for col in [ID_COL]: # проверка на случай, если нам поступил датафрейм с "Идентификатор". Дропаем, т.к. при обучении дропали эту колонку, иначе модель упадет
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [c for c in EXPECTED_COLS if c not in df.columns] # валидация полученного датафрейма. Если колонки будут отличатся, то модель упадет
    if missing:
        return jsonify({"error": "missing columns", "missing": missing}), 400
    df = df[EXPECTED_COLS]

    probs = model.predict_proba(df)[:, 1] # получаем вероятности и предсказания для новых данных
    preds = (probs >= APP_THRESHOLD).astype(int)

    return jsonify({
        'Вероятности': probs.tolist(),
        'Предсказания': preds.tolist()
    }) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # дефолтный порт
