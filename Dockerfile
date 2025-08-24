# Используем официальный Python образ
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p uploaded_pdfs \
    && mkdir -p paper_rag/data/embeddings \
    && mkdir -p paper_rag/data/papers

# Устанавливаем права доступа
RUN chmod +x main.py

# Открываем порт для Streamlit
EXPOSE 8501

# Переменные окружения
ENV PYTHONPATH=/app:/app/ui
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Команда запуска
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
