"""
КОВЧЕГ - Рушій системи
Відповідає за роботу з LLM та RAG
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print("⚠️  llama-cpp-python не встановлено")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("⚠️  sentence-transformers не встановлено")

try:
    import faiss
except ImportError:
    faiss = None
    print("⚠️  faiss-cpu не встановлено")


class KovchegEngine:
    """Основний рушій системи КОВЧЕГ"""
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Ініціалізація рушія
        
        Args:
            config_path: Шлях до файлу конфігурації
        """
        self.config = self._load_config(config_path)
        self.llm = None
        self.embedder = None
        self.loaded_modules = {}  # {назва_модуля: {"index": faiss_index, "texts": [...], "metadata": {...}}}
        
        self._init_embedder()
        self._init_llm()
    
    def _load_config(self, config_path: str) -> dict:
        """Завантаження конфігурації з файлу"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"✅ Конфігурацію завантажено з {config_path}")
            return config
        except Exception as e:
            print(f"❌ Помилка завантаження конфігурації: {e}")
            return {}
    
    def _init_embedder(self):
        """Ініціалізація моделі для створення векторів"""
        if SentenceTransformer is None:
            print("❌ SentenceTransformer недоступний")
            return
        
        try:
            model_path = self.config.get("embedding_settings", {}).get("model_path")
            model_name = self.config.get("embedding_settings", {}).get("model_name")
            
            if model_path and Path(model_path).exists():
                self.embedder = SentenceTransformer(model_path)
                print(f"✅ Модель векторизації завантажено з {model_path}")
            else:
                print(f"⚠️  Локальна модель не знайдена, завантаження {model_name}...")
                self.embedder = SentenceTransformer(model_name)
                print(f"✅ Модель векторизації завантажено: {model_name}")
        except Exception as e:
            print(f"❌ Помилка ініціалізації моделі векторизації: {e}")
    
    def _init_llm(self):
        """Ініціалізація LLM (Qwen)"""
        if Llama is None:
            print("❌ Llama недоступний")
            return
        
        try:
            llm_settings = self.config.get("llm_settings", {})
            model_path = llm_settings.get("model_path")
            
            if not model_path or not Path(model_path).exists():
                print(f"⚠️  Модель LLM не знайдена: {model_path}")
                return
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=llm_settings.get("context_window", 4096),
                n_threads=llm_settings.get("n_threads", 4),
                verbose=False
            )
            print(f"✅ LLM завантажено: {model_path}")
            
        except Exception as e:
            print(f"❌ Помилка ініціалізації LLM: {e}")
    
    def load_module(self, module_name: str, module_path: str) -> bool:
        """
        Завантаження модуля в пам'ять (ліниве завантаження)
        
        Args:
            module_name: Назва модуля
            module_path: Шлях до папки модуля
            
        Returns:
            True якщо успішно завантажено
        """
        if module_name in self.loaded_modules:
            print(f"ℹ️  Модуль '{module_name}' вже завантажено")
            return True
        
        module_path = Path(module_path)
        index_file = module_path / "index.faiss"
        texts_file = module_path / "texts.json"
        manifest_file = module_path / "manifest.json"
        
        # Перевірка наявності необхідних файлів
        if not index_file.exists() or not texts_file.exists():
            print(f"❌ Модуль '{module_name}' пошкоджено: відсутні файли")
            return False
        
        try:
            # Завантаження FAISS індексу
            if faiss is None:
                print("❌ FAISS недоступний")
                return False
            
            index = faiss.read_index(str(index_file))
            
            # Завантаження текстів
            with open(texts_file, "r", encoding="utf-8") as f:
                texts = json.load(f)
            
            # Завантаження метаданих
            metadata = {}
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            
            self.loaded_modules[module_name] = {
                "index": index,
                "texts": texts,
                "metadata": metadata
            }
            
            print(f"✅ Модуль '{module_name}' завантажено ({len(texts)} фрагментів)")
            return True
            
        except Exception as e:
            print(f"❌ Помилка завантаження модуля '{module_name}': {e}")
            return False
    
    def unload_module(self, module_name: str):
        """Вивантаження модуля з пам'яті"""
        if module_name in self.loaded_modules:
            del self.loaded_modules[module_name]
            print(f"ℹ️  Модуль '{module_name}' вивантажено з пам'яті")
    
    def search(self, query: str, module_names: List[str], top_k: int = 5) -> List[Dict]:
        """
        Пошук релевантних фрагментів у завантажених модулях
        
        Args:
            query: Пошуковий запит
            module_names: Список назв модулів для пошуку
            top_k: Кількість результатів
            
        Returns:
            Список знайдених фрагментів з метаданими
        """
        if self.embedder is None:
            print("❌ Модель векторизації недоступна")
            return []
        
        results = []
        
        try:
            # Створення вектора запиту
            query_vector = self.embedder.encode([query])[0]
            query_vector = np.array([query_vector]).astype('float32')
            
            # Пошук у кожному активному модулі
            for module_name in module_names:
                if module_name not in self.loaded_modules:
                    continue
                
                module = self.loaded_modules[module_name]
                index = module["index"]
                texts = module["texts"]
                
                # Пошук найближчих векторів
                distances, indices = index.search(query_vector, min(top_k, len(texts)))
                
                # Формування результатів
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(texts):
                        results.append({
                            "text": texts[idx],
                            "module": module_name,
                            "score": float(1 / (1 + dist)),  # Конвертація відстані в схожість
                            "distance": float(dist)
                        })
            
            # Сортування за релевантністю
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ Помилка пошуку: {e}")
            return []
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Генерація відповіді на основі контексту
        
        Args:
            query: Запит користувача
            context: Контекст з RAG
            
        Returns:
            Згенерована відповідь
        """
        if self.llm is None:
            return "❌ Модель LLM недоступна. Перевірте налаштування."
        
        try:
            llm_settings = self.config.get("llm_settings", {})
            
            # Формування промпту
            prompt = f"""Ти - асистент КОВЧЕГ, який допомагає знаходити інформацію.

Контекст з бази знань:
{context}

Питання користувача: {query}

Дай чітку та корисну відповідь українською мовою на основі наданого контексту. Якщо контекст не містить потрібної інформації, скажи про це чесно."""

            # Генерація відповіді
            response = self.llm(
                prompt,
                max_tokens=llm_settings.get("max_tokens", 512),
                temperature=llm_settings.get("temperature", 0.1),
                top_p=llm_settings.get("top_p", 0.9),
                top_k=llm_settings.get("top_k", 40),
                stop=["Питання:", "Контекст:"],
                echo=False
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            print(f"❌ Помилка генерації відповіді: {e}")
            return f"❌ Помилка генерації відповіді: {e}"
