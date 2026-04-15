#!/usr/bin/env python3
"""
builder.py — Індексуючий модуль для локальної knowledge-base з векторним пошуком.
Читає файли (PDF, DOCX, TXT, MD, JSON, CSV, PY, SH, ZIM),
нарізає текст на оптимальні шматки, векторизує через BGE-M3,
і складає SQLite індекс з dense + sparse векторами + граф знань.
"""
import sys
import os
import sqlite3
import re
import json
import gc
import signal
import hashlib
import logging
import shutil
import atexit
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# НАЛАШТУВАННЯ
# ---------------------------------------------------------------------------
RAW_DIR = Path("modules-raw")
MODULES_DIR = Path("modules")
CHUNK_SIZE = 600
OVERLAP = 100
MIN_CHUNK = 50
MIN_ARTICLE = 100
GRAPH_MIN_W = 2
KW_PER_CHUNK = 20
BATCH_COMMIT = 500
MAX_KW_PAIRS = 200_000

# BGE-M3 налаштування
EMBEDDING_MODEL_PATH = Path("models/BAAI/bge-m3")
VECTOR_DIM = 1024  # BGE-M3 має розмірність 1024
VECTOR_BATCH_SIZE = 32  # Оптимальний розмір батчу для BGE-M3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ОПЦІЙНІ ЗАВИСНОСТІ
# ---------------------------------------------------------------------------
HAS_PDF = False
HAS_DOCX = False
HAS_ZIM = False
HAS_LXML = False
HAS_EMBEDDINGS = False

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    logger.warning("⚠️ pdfplumber не встановлений — PDF пропускатимуться.")

try:
    import docx
    HAS_DOCX = True
except ImportError:
    logger.warning("⚠️ python-docx не встановлений — DOCX пропускатимуться.")

try:
    import libzim
    from bs4 import BeautifulSoup
    HAS_ZIM = True
except ImportError:
    logger.warning("⚠️ libzim або beautifulsoup4 не встановлені — ZIM пропускатимуться.")

try:
    from lxml import etree  # noqa: F401
    HAS_LXML = True
except ImportError:
    pass

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_EMBEDDINGS = True
    logger.info(f"✅ FlagEmbedding (BGE-M3) доступний")
except ImportError:
    logger.error("❌ FlagEmbedding не встановлений! Встановіть: pip install -U FlagEmbedding")
    sys.exit(1)

# ---------------------------------------------------------------------------
# SIGNAL HANDLER
# ---------------------------------------------------------------------------
_active_connection: Optional[sqlite3.Connection] = None
_interrupted = False

def _signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    logger.warning("\n⚠️ Перервано. Завершуємо чисто…")
    if _active_connection:
        try:
            _active_connection.rollback()
            _active_connection.close()
        except Exception:
            pass
    sys.exit(130)

signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# УТИЛІТИ
# ---------------------------------------------------------------------------
def sanitize_module_name(name: str) -> str:
    stripped = name.strip()
    if stripped == "" or stripped in (".", "..") or "/" in stripped or "\\" in stripped:
        raise ValueError(f"Недопустима назва модуля: '{name}'")
    clean = re.sub(r"[^\w\-]", "_", stripped, flags=re.UNICODE)
    if clean == "":
        raise ValueError(f"Недопустима назва модуля: '{name}'")
    return clean

def smart_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    if overlap >= size:
        overlap = size // 4
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            boundary = text.rfind(" ", max(start, end - overlap), end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK:
            chunks.append(chunk)
        start = end if end > start else start + 1
    return chunks

def compute_file_hash(path: Path, algo: str = "sha256", block: int = 65536) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while data := f.read(block):
            h.update(data)
    return h.hexdigest()

def serialize_vector(vec: np.ndarray) -> bytes:
    """Серіалізує numpy вектор у bytes для SQLite"""
    return vec.astype(np.float32).tobytes()

def deserialize_vector(data: bytes) -> np.ndarray:
    """Десеріалізує bytes у numpy вектор"""
    return np.frombuffer(data, dtype=np.float32)

# ---------------------------------------------------------------------------
# ЧИТАЛКИ
# ---------------------------------------------------------------------------
def read_pdf(path: Path) -> Optional[str]:
    if not HAS_PDF:
        return None
    try:
        with pdfplumber.open(path) as p:
            return "\n".join(page.extract_text() or "" for page in p.pages)
    except Exception as e:
        logger.error(f"❌ PDF '{path.name}': {e}")
        return None

def read_docx(path: Path) -> Optional[str]:
    if not HAS_DOCX:
        return None
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.error(f"❌ DOCX '{path.name}': {e}")
        return None

def read_text_file(path: Path) -> Optional[str]:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".sh", ".csv", ".yml", ".yaml", ".xml", ".html", ".htm", ".log"}

def read_file(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in TEXT_EXTENSIONS:
        return read_text_file(path)
    return None

# ---------------------------------------------------------------------------
# КЛАС
# ---------------------------------------------------------------------------
class SQLiteVectorBuilder:
    def __init__(self, module_name: str, description: str):
        self.module_name = sanitize_module_name(module_name)
        self.description = description
        self.module_path = MODULES_DIR / self.module_name
        self.db_path = self.module_path / "search_bge.db"
        self._conn: Optional[sqlite3.Connection] = None
        
        # Ініціалізація BGE-M3
        logger.info(f"🔄 Завантаження BGE-M3 з {EMBEDDING_MODEL_PATH}...")
        
        # Вимикаємо онлайн перевірки
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        if not EMBEDDING_MODEL_PATH.exists():
            logger.error(f"""
            ❌ BGE-M3 модель не знайдена: {EMBEDDING_MODEL_PATH}
            
            Завантажте модель:
            git clone https://huggingface.co/BAAI/bge-m3 {EMBEDDING_MODEL_PATH}
            """)
            sys.exit(1)
        
        try:
            self.model = BGEM3FlagModel(
                str(EMBEDDING_MODEL_PATH),
                use_fp16=False,  # CPU режим
                device='cpu'
            )
            logger.info(f"✅ BGE-M3 завантажена (dim={VECTOR_DIM})")
        except Exception as e:
            logger.error(f"❌ Помилка завантаження BGE-M3: {e}")
            sys.exit(1)

    def _connect(self) -> sqlite3.Connection:
        global _active_connection
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-32768;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        _active_connection = conn
        self._conn = conn
        return conn

    def _close(self):
        global _active_connection
        if self._conn:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            _active_connection = None

    def init_db(self):
        if self.module_path.exists():
            shutil.rmtree(self.module_path)
        self.module_path.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        c = conn.cursor()
        
        # Таблиця з dense та sparse векторами
        c.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                dense_vec BLOB NOT NULL,
                sparse_vec TEXT
            )
        """)
        
        # Індекси
        c.execute("CREATE INDEX IF NOT EXISTS idx_filename ON docs(filename)")
        
        # Граф знань
        c.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                weight INTEGER DEFAULT 1,
                PRIMARY KEY (source, target)
            ) WITHOUT ROWID
        """)
        
        conn.commit()
        return conn

    def encode_batch(self, texts: List[str], show_progress: bool = False) -> Tuple[np.ndarray, List[dict]]:
        """
        Векторизує тексти через BGE-M3
        Повертає: (dense_vectors, sparse_vectors)
        """
        if not texts:
            return np.array([]), []
        
        try:
            # Векторизуємо ВСІ тексти одним викликом
            # BGE-M3 сам розіб'є на батчі
            if show_progress:
                logger.info(f"   🔄 Векторизація {len(texts)} шматків...")
            
            output = self.model.encode(
                texts,
                batch_size=VECTOR_BATCH_SIZE,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            dense_vectors = output['dense_vecs']
            sparse_vectors = output['lexical_weights']
            
            if show_progress:
                logger.info(f"   ✅ Векторизовано успішно")
            
            return dense_vectors, sparse_vectors
            
        except Exception as e:
            logger.error(f"❌ Помилка векторизації: {e}")
            # Fallback: пусті вектори
            empty_dense = np.zeros((len(texts), VECTOR_DIM), dtype=np.float32)
            empty_sparse = [{} for _ in texts]
            return empty_dense, empty_sparse

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        words = re.findall(r"\b[A-Za-zА-Яа-яІіЇїЄєҐґЁё0-9\-]{5,}\b", text)
        seen = set()
        unique = []
        for w in words:
            low = w.lower()
            if low not in seen:
                seen.add(low)
                unique.append(low)
        return unique

    def flush_edges(self, conn: sqlite3.Connection, edges_buffer: Counter):
        if not edges_buffer:
            return
        batch = [(s, t, w) for (s, t), w in edges_buffer.items() if w >= GRAPH_MIN_W]
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO graph_edges (source, target, weight) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
        edges_buffer.clear()

    def insert_chunks_with_embeddings(self, conn: sqlite3.Connection, chunks_data: List[Tuple[str, str]], 
                                     show_progress: bool = False):
        """Вставляє chunks з dense + sparse векторами"""
        if not chunks_data:
            return
        
        filenames, contents = zip(*chunks_data)
        
        # Векторизуємо через BGE-M3
        dense_vectors, sparse_vectors = self.encode_batch(list(contents), show_progress=show_progress)
        
        # Підготовка даних для вставки
        insert_data = []
        for filename, content, dense_vec, sparse_vec in zip(filenames, contents, dense_vectors, sparse_vectors):
            # Конвертуємо sparse vector у JSON-сумісний формат
            sparse_json = None
            if sparse_vec:
                # Конвертуємо numpy.float32 → float для JSON
                sparse_json = json.dumps({
                    str(k): float(v) for k, v in sparse_vec.items()
                })
            
            insert_data.append((
                filename,
                content,
                serialize_vector(dense_vec),
                sparse_json
            ))
        
        # Вставка в базу
        c = conn.cursor()
        c.executemany(
            "INSERT INTO docs (filename, content, dense_vec, sparse_vec) VALUES (?, ?, ?, ?)",
            insert_data
        )
        conn.commit()

    def process_zim(self, file_path: Path, conn: sqlite3.Connection) -> Tuple[int, int]:
        if not HAS_ZIM:
            logger.error("❌ ZIM не підтримується")
            return 0, 0

        logger.info(f"📚 Обробка ZIM: {file_path.name}")
        try:
            archive = libzim.Archive(str(file_path))
            suggestion_searcher = libzim.SuggestionSearcher(archive)
        except Exception as e:
            logger.error(f"❌ Помилка відкриття ZIM/SuggestionSearcher: {e}")
            return 0, 0

        parser = "lxml" if HAS_LXML else "html.parser"

        article_count = 0
        chunk_count = 0
        edges_buffer = Counter()
        seen_titles = set()

        prefixes = ["", "а", "б", "в", "г", "ґ", "д", "е", "є", "ж", "з", "и", "і", "ї", "й",
                    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч",
                    "ш", "щ", "ь", "ю", "я"]

        for prefix in prefixes:
            if _interrupted:
                break

            try:
                suggestion = suggestion_searcher.suggest(prefix)
                estimated = suggestion.getEstimatedMatches()
                logger.info(f"   Prefix '{prefix}' → ≈{estimated:,} пропозицій")

                if estimated == 0:
                    continue

                start = 0
                batch_size = 200
                
                while True:
                    if _interrupted:
                        break

                    try:
                        results = suggestion.getResults(start, batch_size)
                        if not results:
                            break

                        batch_chunks = []  # Chunks для поточного batch
                        
                        for res in results:
                            title = str(res).strip()
                            if not title or title in seen_titles:
                                continue
                            seen_titles.add(title)

                            entry = None
                            candidates = [
                                title,
                                title.replace(" ", "_"),
                                title.replace("_", " "),
                                f"A/{title}",
                                f"A/{title.replace(' ', '_')}",
                                f"A/{title.replace('_', ' ')}",
                            ]

                            for cand in candidates:
                                try:
                                    entry = archive.get_entry_by_title(cand)
                                    break
                                except:
                                    pass
                                try:
                                    entry = archive.get_entry_by_path(cand)
                                    break
                                except:
                                    pass

                            if entry is None:
                                continue

                            if entry.is_redirect:
                                continue

                            item = entry.get_item()
                            if not item.mimetype or "text/html" not in item.mimetype:
                                continue

                            raw = bytes(item.content)
                            soup = BeautifulSoup(raw, parser)
                            for tag in soup(["script", "style", "head", "meta", "nav", "footer", 
                                           "noscript", "iframe", "svg"]):
                                tag.decompose()

                            text = soup.get_text(separator=" ", strip=True)
                            text = re.sub(r"\s+", " ", text).strip()
                            if len(text) < MIN_ARTICLE:
                                continue

                            chunks = smart_chunk(text)
                            full_title = f"Wiki: {title}"

                            for chunk in chunks:
                                # Накопичуємо chunks
                                batch_chunks.append((full_title, chunk))
                                chunk_count += 1

                                # Обробка ключових слів для графу
                                keywords = self.extract_keywords(chunk)[:KW_PER_CHUNK]
                                for a in range(len(keywords)):
                                    for b in range(a + 1, len(keywords)):
                                        pair = tuple(sorted([keywords[a], keywords[b]]))
                                        edges_buffer[pair] += 1

                                if len(edges_buffer) > MAX_KW_PAIRS:
                                    self.flush_edges(conn, edges_buffer)

                            article_count += 1

                            if article_count % 50 == 0:
                                print(f"\r ✅ Статей: {article_count:>5,} | Шматків: {chunk_count:>8,}", end="", flush=True)

                            if article_count % 500 == 0:
                                gc.collect()

                        # Векторизуємо та вставляємо всі chunks з поточного batch
                        if batch_chunks:
                            logger.info(f"   📊 Batch {start}-{start+len(results)}: {len(batch_chunks)} шматків")
                            self.insert_chunks_with_embeddings(conn, batch_chunks, show_progress=True)
                            batch_chunks.clear()

                        if article_count > 0 and article_count % len(results) == 0:
                            start += batch_size
                        else:
                            break

                    except Exception as e:
                        logger.warning(f"Помилка batch для prefix='{prefix}': {e}")
                        break

            except Exception as e:
                logger.warning(f"Помилка suggest для prefix='{prefix}': {e}")
                continue

        self.flush_edges(conn, edges_buffer)
        conn.commit()

        print(f"\n 🎉 ZIM завершено: статей={article_count:,}, шматків={chunk_count:,}")
        logger.info(f"📚 ZIM '{file_path.name}' оброблено.")
        return article_count, chunk_count

    def process_folder(self, source_folder: str):
        source_path = RAW_DIR / source_folder
        if not source_path.is_dir():
            logger.error(f"❌ Папка не існує: {source_path}")
            sys.exit(1)

        conn = self.init_db()
        atexit.register(self._close)

        total_files = 0
        total_chunks = 0
        total_articles = 0
        edges_buffer = Counter()
        file_hashes = {}

        logger.info(f"🚀 Індексація модуля '{self.module_name}' з '{source_path}'…")

        for file_path in sorted(source_path.rglob("*")):
            if _interrupted:
                break
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            ext = file_path.suffix.lower()
            if ext == ".zim":
                articles, chunks = self.process_zim(file_path, conn)
                total_articles += articles
                total_chunks += chunks
                total_files += 1
                try:
                    file_hashes[file_path.name] = compute_file_hash(file_path)
                except Exception:
                    pass
                continue

            logger.info(f"📄 {file_path.name} ({file_path.stat().st_size // 1024} KB)")
            text = read_file(file_path)
            if not text:
                continue

            try:
                file_hashes[file_path.name] = compute_file_hash(file_path)
            except Exception:
                pass

            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < MIN_CHUNK:
                continue

            chunks = smart_chunk(text)
            
            # Збираємо всі chunks для файлу
            file_chunks = [(file_path.name, chunk) for chunk in chunks]
            total_chunks += len(file_chunks)
            
            # Обробка ключових слів для графу
            for chunk in chunks:
                keywords = self.extract_keywords(chunk)[:KW_PER_CHUNK]
                for a in range(len(keywords)):
                    for b in range(a + 1, len(keywords)):
                        pair = tuple(sorted([keywords[a], keywords[b]]))
                        edges_buffer[pair] += 1

                if len(edges_buffer) > MAX_KW_PAIRS:
                    self.flush_edges(conn, edges_buffer)

            # Векторизуємо та вставляємо chunks
            if file_chunks:
                logger.info(f"   📊 {len(file_chunks)} шматків")
                self.insert_chunks_with_embeddings(conn, file_chunks, show_progress=True)

            total_files += 1

        self.flush_edges(conn, edges_buffer)
        conn.commit()

        self._close()

        manifest = {
            "name": self.module_name,
            "description": self.description,
            "created_at": datetime.now().isoformat(),
            "stats": {"files": total_files, "chunks": total_chunks, "articles": total_articles},
            "config": {
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "engine": "sqlite-bge-m3",
                "embedding_model": "BAAI/bge-m3",
                "vector_dim": VECTOR_DIM,
                "supports_sparse": True
            },
            "file_hashes": file_hashes,
        }
        with open(self.module_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info(f"✅ МОДУЛЬ '{self.module_name}' ГОТОВИЙ!")
        logger.info(f" Файлів: {total_files} | Шматків: {total_chunks:,} | Статей ZIM: {total_articles:,}")
        logger.info(f" Розташування: {self.module_path}/")
        logger.info(f" Модель: BGE-M3 (dense + sparse vectors)")
        logger.info("=" * 60)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        logger.error("❌ Потрібно 3 аргументи: назва_модуля опис папка")
        sys.exit(1)

    module_name = sys.argv[1]
    description = sys.argv[2]
    source_folder = sys.argv[3]

    try:
        safe_name = sanitize_module_name(module_name)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if safe_name != module_name:
        logger.warning(f"Назва скоригована: '{module_name}' → '{safe_name}'")

    builder = SQLiteVectorBuilder(safe_name, description)
    builder.process_folder(source_folder)

if __name__ == "__main__":
    main()
