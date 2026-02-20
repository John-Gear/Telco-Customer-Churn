# контейнер как единая точка входа для обучения модели и для запуска flask api
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r -requirements.txt

COPY . .

# учим модель при запуске контейнера один раз, если ранее она не обучалась
RUN chmod +x entrypoint.sh

EXPOSE 5000

CMD ["./entrypoint.sh"]
