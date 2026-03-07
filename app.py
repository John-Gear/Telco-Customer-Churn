from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.logger import get_logger
from src.predict import THRESHOLD, predict_proba, predict

logger = get_logger('api')

app = Flask(__name__)

APP_THRESHOLD = THRESHOLD # забираем порог из predict.py т.к. у нас нет единого env. Принято решение хранить порог только там

logger.info('API started')

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

    try:
        probs = predict_proba(df)  # внутри уже применится cleaner + pipeline(preprocessor + model)
        preds = predict(df, threshold=APP_THRESHOLD)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Prediction failed')
        return jsonify({'error': f'Ошибка инференса: {str(e)}'}), 500

    return jsonify({
        'Вероятности': probs.tolist(),
        'Предсказания': preds.tolist()
    }) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # дефолтный порт
