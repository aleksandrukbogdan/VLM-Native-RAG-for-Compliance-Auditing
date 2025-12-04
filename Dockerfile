# Используем более свежий образ PyTorch 2.3.0
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

# Устанавливаем системные зависимости (poppler нужен обязательно)
# (В этом образе apt может требовать обновления ключей, поэтому || true)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Установка зависимостей Python ---
COPY requirements.txt .

# Важно: удаляем torch из requirements.txt, так как он уже есть в образе
# Иначе pip попытается его переустановить или обновить
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements_no_torch.txt

# --- СКАЧИВАНИЕ МОДЕЛИ ---
COPY download_model.py .
RUN python download_model.py

# Копируем код
COPY . .

# Папки данных
RUN mkdir -p data/input data/output

CMD ["python", "main.py"]
