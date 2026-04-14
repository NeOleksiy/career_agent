# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Устанавливаем Poetry (опционально, если используете)
# Если используете requirements.txt, то закомментируйте этот блок
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем директорию для артефактов данных
RUN mkdir -p ./data_artefacts

# Создаем переменные окружения по умолчанию
ENV API_TOKEN=""
ENV MODEL_URL=""
ENV MODEL_NAME=""
ENV MODEL_TEMP=0.7
ENV MAX_HISTORY=10
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Экспортируем порт для Gradio
EXPOSE 7860

# Команда запуска приложения
CMD ["python", "app_gradio.py"]