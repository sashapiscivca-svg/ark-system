import streamlit as st
import sqlite3
from pathlib import Path
import re
import numpy as np
from functools import lru_cache
import os
from concurrent.futures import ThreadPoolExecutor

# --- НАЛАШТУВАННЯ ---
st.set_page_config(
    page_title="КОВЧЕГ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- СТИЛІ ---
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; }
    .source-box {
        background: #1e1e1e;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #00AEEF;
    }
    .ai-response {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #00AEEF;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .score-badge {
        display: inline-block;
        background: #00AEEF;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- КОНФІГУРАЦІЯ ---
@st.cache_resource
def get_config():
    env_root = os.environ.get('KOVCHEG_ROOT')
    base = Path(env_root) if env_root else Path.cwd()
    
    llm_model = base / "models" / "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    embedding_model_path = base / "models" / "BAAI" / "bge-m3"
    reranker_model_path = base / "models" / "BAAI" / "bge-reranker-v2-m3"
    
    return base, base / "modules", llm_model, embedding_model_path, reranker_model_path

BASE_DIR, MODULES_DIR, MODEL_PATH, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH = get_config()

# --- ПАРАМЕТРИ ГЕНЕРАЦІЇ ---
GEN_SETTINGS = {
    "n_ctx": 8192,
    "n_batch": 512,
    "max_tokens": 600,
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.15,
    "threads": os.cpu_count(),
    "top_k": 30,
    "n_gpu_layers": 0
}

# --- ВЕКТОРНІ УТИЛІТИ ---
def serialize_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def deserialize_vector(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# --- ОСНОВНИЙ КЛАС ---
@st.cache_resource
class ArkEngine:
    def __init__(self):
        self.llm = None
        self.llm_loaded = False
        self.embedding_model = None
        self.reranker_model = None
        self._load_embedding_model()
        self._load_reranker_model()
        if MODEL_PATH and MODEL_PATH.exists():
            self._load_llm()

    def _load_embedding_model(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            if not EMBEDDING_MODEL_PATH.exists():
                st.error(f"❌ Модель пошуку не знайдена за адресою: {EMBEDDING_MODEL_PATH}")
                return False
            
            self.embedding_model = BGEM3FlagModel(
                str(EMBEDDING_MODEL_PATH),
                use_fp16=False,
                device='cpu'
            )
            return True
            
        except Exception as e:
            st.error(f"❌ Помилка завантаження пошуку: {e}")
            return False

    def _load_reranker_model(self):
        try:
            from FlagEmbedding import FlagReranker
            
            if not RERANKER_MODEL_PATH.exists():
                st.sidebar.info("ℹ️ Працюємо без додаткової перевірки релевантності")
                return False
            
            self.reranker_model = FlagReranker(
                str(RERANKER_MODEL_PATH),
                use_fp16=False,
                device='cpu'
            )
            return True
            
        except Exception as e:
            st.sidebar.info("ℹ️ Працюємо в базовому режимі пошуку")
            return False

    def _load_llm(self):
        if self.llm_loaded:
            return True
        if not MODEL_PATH or not MODEL_PATH.exists():
            return False
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=GEN_SETTINGS["n_ctx"],
                n_batch=GEN_SETTINGS["n_batch"],
                n_threads=GEN_SETTINGS["threads"],
                n_gpu_layers=GEN_SETTINGS["n_gpu_layers"],
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            _ = self.llm("Тест", max_tokens=5, temperature=0.0)
            self.llm_loaded = True
            return True
        except Exception as e:
            st.error(f"❌ Не вдалося завантажити AI асистента: {e}")
            return False

    def get_db_path(self, mod_name):
        return MODULES_DIR / mod_name / "search_bge.db"

    def init_module_db(self, mod_name):
        path = self.get_db_path(mod_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                dense_vec BLOB NOT NULL,
                sparse_vec TEXT
            )
        """)
        
        c.execute("CREATE INDEX IF NOT EXISTS idx_filename ON docs(filename)")
        
        conn.commit()
        conn.close()

    def encode_text(self, text: str, return_dense=True, return_sparse=False):
        if not self.embedding_model:
            return None
        
        try:
            output = self.embedding_model.encode(
                [text],
                batch_size=1,
                max_length=8192,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False
            )
            
            if return_dense and return_sparse:
                return output['dense_vecs'][0], output['lexical_weights'][0]
            elif return_dense:
                return output['dense_vecs'][0]
            elif return_sparse:
                return output['lexical_weights'][0]
            
        except Exception as e:
            st.error(f"Помилка обробки запиту: {e}")
            return None

    def search_vector(self, query, active_mods, limit=15, use_reranker=True):
        if not active_mods or not self.embedding_model:
            return []
        
        initial_limit = 50 if use_reranker and self.reranker_model else limit
        
        query_vec = self.encode_text(query, return_dense=True, return_sparse=False)
        if query_vec is None:
            return []
        
        results = []
        
        def search_module(mod):
            db_path = self.get_db_path(mod)
            if not db_path.exists():
                return []
            
            try:
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                c = conn.cursor()
                c.execute("SELECT id, filename, content, dense_vec FROM docs")
                rows = c.fetchall()
                conn.close()
                
                if not rows:
                    return []
                
                similarities = []
                for doc_id, filename, content, dense_bytes in rows:
                    doc_vec = deserialize_vector(dense_bytes)
                    similarity = cosine_similarity(query_vec, doc_vec)
                    similarities.append({
                        "id": doc_id,
                        "source": filename,
                        "text": content,
                        "score": float(similarity),
                        "module": mod,
                        "rerank_score": None
                    })
                
                return sorted(similarities, key=lambda x: x["score"], reverse=True)[:initial_limit * 2]
                
            except Exception as e:
                st.error(f"Помилка пошуку в {mod}: {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            all_results = list(executor.map(search_module, active_mods))
        
        for mod_results in all_results:
            results.extend(mod_results)
        
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:initial_limit]
        
        if use_reranker and self.reranker_model and results:
            try:
                pairs = [[query, r["text"]] for r in results]
                rerank_scores = self.reranker_model.compute_score(pairs, batch_size=32, max_length=1024)
                
                for i, score in enumerate(rerank_scores):
                    results[i]["rerank_score"] = float(score)
                
                results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
                
            except Exception as e:
                st.warning(f"Додаткова перевірка пропущена: {e}")
        
        return results[:limit]

    def generate_answer(self, query, context):
        if not self._load_llm() or not context:
            yield "ℹ️ Не вдалося згенерувати відповідь"
            return

        top_context = context[:10]
        
        ctx_str = "\n\n".join([
            f"[Джерело {i+1}: {c['source']}]\n{c['text']}"
            for i, c in enumerate(top_context)
        ])

        system_prompt = """Ти — аналітичний помічник КОВЧЕГ. Твоя мета — точність і фактаж.
Відповідай виключно на основі наданих фрагментів (Context).

ПРАВИЛА:
1. ЦИТУЙ ТОЧНО: Медичні терміни, діагнози, класифікації (злоякісний/доброякісний), цифри та імена зберігай як у джерелі. Не змінюй їх заради спрощення.
2. СТРУКТУРА: Якщо джерел декілька, збери інформацію в логічну відповідь. Використовуй списки для перерахування.
3. БЕЗПЕКА: Якщо в тексті є протиріччя (наприклад, "доброякісний" і "злоякісний" поруч), вкажи обидва варіанти або повідом про невизначеність.
4. МОВА: Відповідай українською мовою.

ЗАБОРОНИ:
• Ніколи не вигадуй факти.
• Не використовуй власні знання, якщо вони суперечать контексту.
• Не роби висновків, яких немає в тексті.

Якщо інформації недостатньо для повної відповіді, напиши: "Інформація в наданих документах відсутня або неповна".
Використовуй тільки українську кирилицю. Не вставляй слова іноземними мовами або латиницею."""

        full_prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
ІНФОРМАЦІЯ З ДОКУМЕНТІВ:
{ctx_str}

ПИТАННЯ: {query}

Дай відповідь на основі цих документів:<|end|>
<|assistant|>
"""

        stream = self.llm(
            full_prompt,
            max_tokens=GEN_SETTINGS["max_tokens"],
            temperature=GEN_SETTINGS["temperature"],
            top_p=GEN_SETTINGS["top_p"],
            top_k=GEN_SETTINGS["top_k"],
            repeat_penalty=GEN_SETTINGS["repeat_penalty"],
            stop=["<|end|>", "<|endoftext|>", "<|user|>"],
            stream=True,
        )

        response = ""
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            response += token
            yield response

# Ініціалізація
engine = ArkEngine()

# --- БІЧНА ПАНЕЛЬ ---
with st.sidebar:
    st.header("📚 Бази знань")
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    mods = sorted([d.name for d in MODULES_DIR.iterdir() if d.is_dir()])

    if "active_mods" not in st.session_state:
        st.session_state.active_mods = set()

    if not mods:
        st.info("Поки немає жодної бази знань")
    else:
        for m in mods:
            db_path = MODULES_DIR / m / "search_bge.db"
            has_data = False
            doc_count = 0
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM docs")
                    doc_count = c.fetchone()[0]
                    conn.close()
                    has_data = doc_count > 0
                except:
                    pass
            
            icon = "✅" if has_data else "📂"
            label = f"{icon} {m}"
            
            if st.checkbox(label, value=m in st.session_state.active_mods, key=f"mod_{m}"):
                st.session_state.active_mods.add(m)
            else:
                st.session_state.active_mods.discard(m)

    st.divider()
    st.subheader("⚙️ Налаштування")
    use_ai = st.toggle("🤖 Відповідь від AI", value=True, help="Отримати готову відповідь замість лише списку джерел")
    use_reranker = st.toggle("🎯 Точний пошук", value=True, help="Додаткова перевірка релевантності результатів")
    show_scores = st.toggle("📊 Показати оцінки", value=True)
    top_k = st.slider("Скільки результатів показати", 5, 20, 10)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Оновити", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    with st.expander("❓ Довідка"):
        st.markdown("""
**Що вміє система:**
- Шукає інформацію у ваших документах
- Дає відповіді на основі знайденого
- Показує джерела інформації

**Як користуватись:**
1. Оберіть бази знань зліва
2. Введіть своє питання
3. Отримайте відповідь з посиланнями на джерела

**Корисно знати:**
- Система відповідає лише на основі ваших документів
- Якщо інформації немає — чесно про це каже
- Можна переглянути всі джерела відповіді
        """)

# --- ГОЛОВНА ЧАСТИНА ---
st.title("🛡️ КОВЧЕГ")
st.caption("Локальний пошук по документах та ШІ-відповіді")

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_sources(sources, show_scores=True):
    shown = set()
    for src in sources:
        key = f"{src['source']}_{src['text'][:50]}"
        if key in shown:
            continue
        shown.add(key)
        
        badges = ""
        if show_scores:
            emb_score = int(src['score'] * 100)
            badges += f'<span class="score-badge">Релевантність: {emb_score}%</span>'
            
            if src.get('rerank_score') is not None:
                rerank_display = int(min(src['rerank_score'] * 100, 100))
                badges += f'<span class="score-badge">Перевірено: {rerank_display}%</span>'
        
        st.markdown(f"""
<div class="source-box">
    <strong>📄 {src['source']}</strong> <small>({src.get('module', 'N/A')})</small>{badges}<br>
    <small>{src['text'][:350]}...</small>
</div>
""", unsafe_allow_html=True)

# Історія чату
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            if msg.get("content"):
                st.markdown(f'<div class="ai-response">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📚 Звідки ця інформація ({len(msg['sources'])} джерел)"):
                    render_sources(msg["sources"], show_scores)

# --- ДОДАВАННЯ ФАЙЛІВ ---
with st.popover("📎 Додати документи", use_container_width=True):
    if not mods:
        st.info("Створіть першу базу знань")
        new_mod = st.text_input("Назва бази знань:")
        if st.button("Створити") and new_mod:
            sanitized = re.sub(r'[^\w\-]', '_', new_mod.strip())
            (MODULES_DIR / sanitized).mkdir(parents=True, exist_ok=True)
            st.success(f"База '{sanitized}' створена!")
            st.rerun()
    else:
        st.info("💡 Використовуйте builder.py для додавання документів до бази")

# --- ОБРОБКА ЗАПИТІВ ---
if prompt := st.chat_input("Поставте своє питання..."):
    if not st.session_state.active_mods:
        st.error("❌ Оберіть хоча б одну базу знань!")
    elif not engine.embedding_model:
        st.error("❌ Пошукова система не готова")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            search_msg = "🔍 Шукаю відповідь"
            if use_reranker and engine.reranker_model:
                search_msg = "🔍 Шукаю та перевіряю результати"
            
            with st.spinner(search_msg + "..."):
                results = engine.search_vector(
                    prompt, 
                    list(st.session_state.active_mods), 
                    limit=top_k,
                    use_reranker=use_reranker
                )

            if not results:
                st.warning("🔍 На жаль, не знайшов відповіді у вибраних базах знань")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Вибачте, не знайшов потрібної інформації у вибраних базах.",
                    "sources": []
                })
            else:
                final_answer = ""
                if use_ai and engine.llm_loaded:
                    placeholder = st.empty()
                    for chunk in engine.generate_answer(prompt, results):
                        final_answer = chunk
                        placeholder.markdown(
                            f'<div class="ai-response">{final_answer}▌</div>', 
                            unsafe_allow_html=True
                        )
                    placeholder.markdown(
                        f'<div class="ai-response">{final_answer}</div>', 
                        unsafe_allow_html=True
                    )
                elif use_ai and not engine.llm_loaded:
                    st.info("🤖 AI асистент недоступний — показую знайдені документи")
                else:
                    st.info("🤖 Режим AI відповідей вимкнено")

                with st.expander(f"📚 Знайдені джерела ({len(results)})", expanded=not use_ai):
                    render_sources(results, show_scores)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer if (use_ai and final_answer) else None,
                    "sources": results
                })
