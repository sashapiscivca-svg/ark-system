# КОВЧЕГ

Локальна RAG-система для пошуку по документах з AI-відповідями. Працює повністю офлайн.

## Принцип роботи

1. Документи (PDF, DOCX, TXT, ZIM та ін.) індексуються через `builder.py` — текст нарізається на фрагменти та векторизується моделлю BGE-M3
2. Векторний індекс зберігається локально в SQLite
3. При запиті система знаходить найрелевантніші фрагменти (dense + reranker)
4. LLM (Phi-3.5-mini або будь-яка GGUF модель) формує відповідь виключно на основі знайдених документів

## Встановлення

```bash
git clone <repo-url>
cd ark-system

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Моделі

Завантажити і покласти в `models/`:

**LLM** (будь-яка GGUF):
```bash
# Приклад — Phi-3.5-mini (2.4 ГБ)
wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf -P models/
```

**Векторизація** (BGE-M3):
```bash
git clone https://huggingface.co/BAAI/bge-m3 models/BAAI/bge-m3
```

**Ре-ранкер** (опційно, покращує точність):
```bash
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3 models/BAAI/bge-reranker-v2-m3
```

## Генерація ключів

Ключі потрібні для підпису модулів знань. Кожен запуск — свої ключі.

```bash
python scripts/keygen.py
```

Приватний ключ (`keys/private_key.pem`) — не передавати нікому, не додавати до git.

## Індексація документів

```bash
python scripts/builder.py <назва_модуля> "<опис>" <папка_в_modules-raw>
```

Приклад:
```bash
# Покласти документи в modules-raw/medicine/
python scripts/builder.py medicine "Медична база знань" medicine
```

Підтримувані формати: PDF, DOCX, TXT, MD, JSON, CSV, PY, SH, ZIM

## Запуск

```bash
streamlit run app.py
```

Або як скомпільований застосунок:
```bash
python run.py
```

## Збірка портативної версії

```bash
pip install pyinstaller
pyinstaller build.spec
```

Результат у `build/KOVCHEG/` — можна запускати з USB.

## Структура проєкту

```
ark-system/
├── app.py              # Streamlit інтерфейс
├── backend.py          # Рушій (legacy — faiss + sentence-transformers)
├── security.py         # Перевірка підписів модулів
├── run.py              # Точка входу для збірки PyInstaller
├── run.sh              # Shell-запуск
├── build.spec          # Конфіг PyInstaller
├── config.json         # Налаштування системи
├── requirements.txt    # Залежності
├── scripts/
│   ├── builder.py      # Індексація документів
│   ├── keygen.py       # Генерація ключів Ed25519
│   ├── setup.py        # Початкове налаштування
│   └── check_db.py     # Перевірка бази знань
├── models/             # Моделі (не в git — завантажити окремо)
├── modules/            # Індекси БЗ (генеруються локально)
├── modules-raw/        # Вхідні документи (не в git)
└── keys/               # Ключі Ed25519 (не в git — генерувати локально)
```

## Залежності

| Пакет | Для чого |
|-------|----------|
| streamlit | Веб-інтерфейс |
| llama-cpp-python | Запуск LLM (.gguf) |
| FlagEmbedding | Векторизація BGE-M3 + ре-ранкер |
| numpy | Векторні обчислення |
| pdfplumber | Читання PDF |
| python-docx | Читання DOCX |
| beautifulsoup4 + libzim | Читання ZIM (Wikipedia offline) |
| cryptography | Підписи модулів Ed25519 |
