#!/bin/sh
set -e

# если модель еще не обучена, то контейнер запустится только после того как модель отдаст веса model.joblib 
if [ ! -f artifacts/model.joblib ]; then
  echo "Model not found. Training..."
  python -m src.train
fi

echo "Starting API..."
python app.py