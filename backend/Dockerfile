# Используем официальный Python образ
FROM python:3.9-slim

# Обновляем pip и устанавливаем зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем код приложения и pkl-файл ретривера
COPY . .

# Если у вас есть статические файлы (например, index.html), можно их положить в папку static
RUN mkdir -p static
# (Либо frontend будет обслуживаться отдельно)

EXPOSE 5000
CMD ["python", "app.py"]