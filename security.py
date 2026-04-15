"""
КОВЧЕГ - Модуль безпеки
Відповідає за перевірку цифрових підписів модулів
"""

import os
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature


class SecurityManager:
    """Менеджер безпеки для перевірки підписів модулів"""
    
    def __init__(self, keys_path: str = "./keys"):
        """
        Ініціалізація менеджера безпеки
        
        Args:
            keys_path: Шлях до папки з ключами
        """
        self.keys_path = Path(keys_path)
        self.public_key = None
        self._load_public_key()
    
    def _load_public_key(self):
        """Завантаження публічного ключа з файлу"""
        public_key_file = self.keys_path / "public_key.pem"
        
        if not public_key_file.exists():
            print(f"⚠️  УВАГА: Публічний ключ не знайдено в {public_key_file}")
            return
        
        try:
            with open(public_key_file, "rb") as f:
                self.public_key = serialization.load_pem_public_key(f.read())
            print(f"✅ Публічний ключ завантажено з {public_key_file}")
        except Exception as e:
            print(f"❌ Помилка завантаження публічного ключа: {e}")
    
    def verify_signature(self, module_path: str) -> bool:
        """
        Перевірка цифрового підпису модуля
        
        Args:
            module_path: Шлях до папки модуля
            
        Returns:
            True якщо підпис валідний, False інакше
        """
        module_path = Path(module_path)
        
        # Перевірка наявності публічного ключа
        if self.public_key is None:
            print(f"❌ Неможливо перевірити підпис: публічний ключ не завантажено")
            return False
        
        # Перевірка наявності файлу підпису
        signature_file = module_path / "manifest.sig"
        if not signature_file.exists():
            print(f"❌ Файл підпису не знайдено: {signature_file}")
            return False
        
        # Перевірка наявності маніфесту
        manifest_file = module_path / "manifest.json"
        if not manifest_file.exists():
            print(f"❌ Маніфест не знайдено: {manifest_file}")
            return False
        
        try:
            # Читання підпису
            with open(signature_file, "rb") as f:
                signature = f.read()
            
            # Читання даних маніфесту
            with open(manifest_file, "rb") as f:
                manifest_data = f.read()
            
            # Перевірка підпису
            self.public_key.verify(signature, manifest_data)
            print(f"✅ Підпис модуля '{module_path.name}' валідний")
            return True
            
        except InvalidSignature:
            print(f"❌ ПОШКОДЖЕНО: Невалідний підпис модуля '{module_path.name}'")
            return False
        except Exception as e:
            print(f"❌ Помилка перевірки підпису модуля '{module_path.name}': {e}")
            return False
    
    def get_module_info(self, module_path: str) -> dict:
        """
        Отримання інформації про модуль з маніфесту
        
        Args:
            module_path: Шлях до папки модуля
            
        Returns:
            Словник з інформацією про модуль
        """
        module_path = Path(module_path)
        manifest_file = module_path / "manifest.json"
        
        if not manifest_file.exists():
            return {
                "name": module_path.name,
                "description": "Маніфест не знайдено",
                "verified": False
            }
        
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            manifest["verified"] = self.verify_signature(module_path)
            return manifest
            
        except Exception as e:
            return {
                "name": module_path.name,
                "description": f"Помилка читання маніфесту: {e}",
                "verified": False
            }
