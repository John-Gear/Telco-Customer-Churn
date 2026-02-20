from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.logger import get_logger
from predict import THRESHOLD

logger = get_logger('api')
logger.info('API started, model load')

app = Flask(__name__)

MODEL_PATH = 'artifacts/model.joblib'
ID_COL = 'Идентификатор'
APP_THRESHOLD = THRESHOLD # забираем порог из predict.py т.к. у нас нет единого env. Принято решение хранить порог только там

# один раз загружаем модель при старте сервера
model = joblib.load(MODEL_PATH)

# проверка жив ли сервер
@app.get('/health')
def health():
    return jsonify({'status': 'ok'})\

# берем json из тела запроса
@app.post('/predict')
def predict():
    payload = request.get_json()
    '''

    тело запроса

    '''
    if isinstance(payload, dict): # можем принять либо 1 клиента, либо список клиентов
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return jsonify({'error'}), 400
    
    df = pd.DataFrame(rows)

    for col in [ID_COL]: # проверка на случай, если нам поступил датафрейм с "Идентификатор". Дропаем, т.к. при обучении дропали эту колонку, иначе модель упадет
        if col in df.columns:
            df = df.drop(columns=[col])

    probs = model.predict_proba(df)[:, 1] # получаем вероятности и предсказания для новых данных
    preds = (probs >= APP_THRESHOLD)

    return jsonify({
        'Вероятности': probs.tolist(),
        'Предсказания': preds.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
