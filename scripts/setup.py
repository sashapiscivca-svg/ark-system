#!/usr/bin/env python3
"""
КОВЧЕГ - Скрипт початкового налаштування
Завантажує моделі та створює необхідні файли конфігурації
"""

import os
import sys
import json
from pathlib import Path
from urllib.request import urlretrieve


def download_with_progress(url: str, filepath: str):
    """
    Завантаження файлу з індикатором прогресу
    
    Args:
        url: URL для завантаження
        filepath: Шлях для збереження
    """
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r⏳ Завантаження: {percent}%")
        sys.stdout.flush()
    
    try:
        urlretrieve(url, filepath, progress_hook)
        sys.stdout.write("\r✅ Завантаження завершено!" + " " * 20 + "\n")
        return True
    except Exception as e:
        sys.stdout.write("\r❌ Помилка завантаження" + " " * 20 + "\n")
        print(f"   Деталі: {e}")
        return False


def setup_directories():
    """Створення необхідних директорій"""
    print("\n📁 Створюю структуру папок...")
    
    directories = [
        "./models",
        "./modules",
        "./keys",
        "./scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True


def create_default_config():
    """Створення дефолтного файлу конфігурації"""
    config_file = Path("./config.json")
    
    if config_file.exists():
        print("\n⚠️  Файл config.json вже існує")
        response = input("   Перезаписати? (yes/no): ").strip().lower()
        if response not in ["yes", "y", "так", "т"]:
            print("   ℹ️  Залишаю існуючий config.json")
            return True
    
    print("\n⚙️  Створюю config.json...")
    
    default_config = {
        "llm_settings": {
            "model_path": "./models/qwen2.5-3b-instruct-q4_k_m.gguf",
            "context_window": 4096,
            "temperature": 0.1,
            "n_threads": 4,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 40
        },
        "embedding_settings": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_path": "./models/all-MiniLM-L6-v2"
        },
        "paths": {
            "modules": "./modules",
            "keys": "./keys",
            "models": "./models"
        },
        "ui_settings": {
            "page_title": "КОВЧЕГ - Локальна RAG-система",
            "page_icon": "🛡️",
            "header": "КОВЧЕГ",
            "description": "Безпечна локальна система пошуку знань"
        },
        "rag_settings": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k_results": 5
        }
    }
    
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print("   ✅ config.json створено")
        return True
    except Exception as e:
        print(f"   ❌ Помилка створення config.json: {e}")
        return False


def download_llm_model():
    """Завантаження моделі Qwen"""
    print("\n🤖 Завантаження LLM моделі (Qwen2.5-3B-Instruct)...")
    
    model_path = Path("./models/qwen2.5-3b-instruct-q4_k_m.gguf")
    
    if model_path.exists():
        print("   ℹ️  Модель вже існує, пропускаю завантаження")
        return True
    
    # URL для завантаження моделі Qwen з Hugging Face
    model_url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    print(f"   📥 URL: {model_url}")
    print(f"   📁 Зберігаю в: {model_path}")
    print("   ⚠️  УВАГА: Розмір файлу ~2GB, завантаження може зайняти час...")
    
    response = input("\n   Продовжити завантаження? (yes/no): ").strip().lower()
    if response not in ["yes", "y", "так", "т"]:
        print("   ⏭️  Пропускаю завантаження LLM")
        return False
    
    return download_with_progress(model_url, str(model_path))


def download_embedding_model():
    """Завантаження моделі векторизації"""
    print("\n🧠 Завантаження моделі векторизації (all-MiniLM-L6-v2)...")
    
    model_path = Path("./models/all-MiniLM-L6-v2")
    
    if model_path.exists():
        print("   ℹ️  Модель вже існує, пропускаю завантаження")
        return True
    
    print("   📥 Завантаження через sentence-transformers...")
    print("   ⚠️  Перший запуск може зайняти кілька хвилин...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Завантаження моделі
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Збереження локально
        model.save(str(model_path))
        print(f"   ✅ Модель збережено в {model_path}")
        return True
        
    except ImportError:
        print("   ❌ sentence-transformers не встановлено")
        print("   Виконайте: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"   ❌ Помилка завантаження моделі: {e}")
        return False


def create_readme():
    """Створення файлу README"""
    readme_file = Path("./README.md")
    
    if readme_file.exists():
        return True
    
    print("\n📝 Створюю README.md...")
    
    readme_content = """# КОВЧЕГ - Локальна RAG-система

Безпечна локальна система пошуку знань з підтримкою цифрових підписів.

## 🚀 Швидкий старт

### 1. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 2. Початкове налаштування
```bash
python scripts/setup.py
```

### 3. Генерація ключів (опціонально)
```bash
python scripts/keygen.py
```

### 4. Створення модуля знань
```bash
python scripts/builder.py <файл.txt> <назва_модуля> "Опис модуля"
```

Приклад:
```bash
python scripts/builder.py documents/manual.txt user_manual "Посібник користувача"
```

### 5. Запуск інтерфейсу
```bash
streamlit run app.py
```

## 📁 Структура проєкту

```
kovcheg/
├── app.py                 # Веб-інтерфейс
├── backend.py            # Рушій системи
├── security.py           # Модуль безпеки
├── config.json           # Конфігурація
├── requirements.txt      # Залежності
├── scripts/              # Скрипти керування
│   ├── setup.py         # Початкове налаштування
│   ├── keygen.py        # Генерація ключів
│   └── builder.py       # Створення модулів
├── models/              # LLM та embedding моделі
├── modules/             # Модулі бази знань
└── keys/                # Криптографічні ключі
```

## 🎨 Особливості

- ✅ Підтримка темної та світлої теми
- ✅ Цифрові підписи модулів (Ed25519)
- ✅ Ліниве завантаження модулів
- ✅ Локальна робота (без інтернету)
- ✅ FAISS векторний пошук
- ✅ LLM на базі Qwen2.5

## ⚙️ Налаштування

Усі параметри налаштовуються через `config.json`:
- Параметри LLM (температура, контекст)
- Шляхи до моделей та даних
- Налаштування RAG (розмір фрагментів, перекриття)
- Інтерфейс користувача

## 🔐 Безпека

- Усі модулі мають цифровий підпис
- Пошкоджені модулі автоматично блокуються
- Приватний ключ зберігається локально
- Алгоритм підпису: Ed25519

## 📖 Документація

Детальну документацію дивіться у файлах проєкту.

## ⚖️ Ліцензія

Проєкт створено для освітніх цілей.
"""
    
    try:
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("   ✅ README.md створено")
        return True
    except Exception as e:
        print(f"   ❌ Помилка створення README.md: {e}")
        return False


def main():
    """Головна функція"""
    print("=" * 60)
    print("🛡️  КОВЧЕГ - Початкове налаштування системи")
    print("=" * 60)
    
    # Крок 1: Створення директорій
    if not setup_directories():
        print("\n❌ Помилка створення директорій")
        sys.exit(1)
    
    # Крок 2: Створення конфігурації
    if not create_default_config():
        print("\n❌ Помилка створення конфігурації")
        sys.exit(1)
    
    # Крок 3: Створення README
    create_readme()
    
    # Крок 4: Завантаження моделі векторизації
    print("\n" + "=" * 60)
    print("📦 Завантаження моделей")
    print("=" * 60)
    
    embedding_success = download_embedding_model()
    
    # Крок 5: Завантаження LLM моделі (опціонально)
    llm_success = download_llm_model()
    
    # Фінальний звіт
    print("\n" + "=" * 60)
    print("📊 Підсумок налаштування")
    print("=" * 60)
    
    print(f"\n✅ Структура папок створена")
    print(f"✅ Конфігурація створена")
    print(f"{'✅' if embedding_success else '⚠️ '} Модель векторизації: {'Готова' if embedding_success else 'Не завантажена'}")
    print(f"{'✅' if llm_success else '⚠️ '} LLM модель: {'Готова' if llm_success else 'Не завантажена'}")
    
    print("\n" + "=" * 60)
    print("🎯 Наступні кроки:")
    print("=" * 60)
    
    if not llm_success:
        print("\n⚠️  LLM модель не завантажена!")
        print("   Опції:")
        print("   1. Повторити налаштування: python scripts/setup.py")
        print("   2. Завантажити модель вручну з Hugging Face")
        print("   3. Використати іншу GGUF модель (оновіть config.json)")
    
    print("\n📝 Рекомендовані дії:")
    print("   1. Згенеруйте ключі: python scripts/keygen.py")
    print("   2. Створіть модуль: python scripts/builder.py <файл> <назва>")
    print("   3. Запустіть систему: streamlit run app.py")
    
    print("\n📚 Документація:")
    print("   README.md - Повна документація проєкту")
    
    print("\n" + "=" * 60)
    print("✅ Налаштування завершено!")
    print("=" * 60)


if __name__ == "__main__":
    main()
