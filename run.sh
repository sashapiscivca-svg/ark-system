#!/bin/bash

# КОВЧЕГ - Скрипт запуску
# Автоматичний запуск веб-інтерфейсу

echo "🛡️  КОВЧЕГ - Система локального пошуку знань"
echo "=============================================="
echo ""

# Перевірка наявності Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не знайдено. Встановіть Python 3.8 або новіше."
    exit 1
fi

# Перевірка наявності конфігурації
if [ ! -f "config.json" ]; then
    echo "⚠️  Конфігурація не знайдена. Запускаю початкове налаштування..."
    python3 scripts/setup.py
    if [ $? -ne 0 ]; then
        echo "❌ Помилка налаштування"
        exit 1
    fi
fi

# Перевірка встановлення залежностей
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️  Залежності не встановлені. Встановлюю..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Помилка встановлення залежностей"
        exit 1
    fi
fi

# Запуск Streamlit
echo "🚀 Запускаю КОВЧЕГ..."
echo ""
echo "📱 Інтерфейс буде доступний за адресою: http://localhost:8501"
echo "⏹️  Для зупинки натисніть Ctrl+C"
echo ""

streamlit run app.py
