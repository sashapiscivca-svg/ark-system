#!/usr/bin/env python3
"""
КОВЧЕГ - Генератор ключів
Створює пару криптографічних ключів Ed25519
"""

import sys
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keys(keys_path: str = "./keys"):
    """
    Генерація пари ключів Ed25519
    
    Args:
        keys_path: Шлях до папки для збереження ключів
    """
    keys_path = Path(keys_path)
    keys_path.mkdir(parents=True, exist_ok=True)
    
    print("🔐 КОВЧЕГ - Генератор ключів")
    print("=" * 50)
    
    # Перевірка чи існують ключі
    private_key_file = keys_path / "private_key.pem"
    public_key_file = keys_path / "public_key.pem"
    
    if private_key_file.exists() or public_key_file.exists():
        print("\n⚠️  УВАГА: Ключі вже існують!")
        print(f"   Приватний ключ: {private_key_file}")
        print(f"   Публічний ключ: {public_key_file}")
        
        response = input("\nПерезаписати існуючі ключі? (yes/no): ").strip().lower()
        if response not in ["yes", "y", "так", "т"]:
            print("\n❌ Операцію скасовано")
            return False
        
        print("\n⚠️  Видаляю старі ключі...")
    
    try:
        # Генерація приватного ключа
        print("\n🔑 Генерую приватний ключ...")
        private_key = Ed25519PrivateKey.generate()
        
        # Збереження приватного ключа
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(private_key_file, "wb") as f:
            f.write(private_pem)
        print(f"✅ Приватний ключ збережено: {private_key_file}")
        
        # Генерація та збереження публічного ключа
        print("\n🔓 Генерую публічний ключ...")
        public_key = private_key.public_key()
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open(public_key_file, "wb") as f:
            f.write(public_pem)
        print(f"✅ Публічний ключ збережено: {public_key_file}")
        
        # Встановлення прав доступу (тільки для власника)
        try:
            import os
            os.chmod(private_key_file, 0o600)  # rw-------
            os.chmod(public_key_file, 0o644)   # rw-r--r--
            print("\n🔒 Права доступу налаштовано")
        except Exception as e:
            print(f"\n⚠️  Не вдалося налаштувати права доступу: {e}")
        
        print("\n" + "=" * 50)
        print("✅ УСПІХ! Ключі успішно згенеровано")
        print("=" * 50)
        print("\n⚠️  ВАЖЛИВО:")
        print("   • Приватний ключ - СТРОГО КОНФІДЕНЦІЙНИЙ!")
        print("   • Не передавайте приватний ключ нікому")
        print("   • Зробіть резервну копію ключів у безпечному місці")
        print("   • Публічний ключ використовується для перевірки підписів")
        print("\n📁 Розташування:")
        print(f"   Приватний: {private_key_file.absolute()}")
        print(f"   Публічний: {public_key_file.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА при генерації ключів: {e}")
        return False


def main():
    """Головна функція"""
    if len(sys.argv) > 1:
        keys_path = sys.argv[1]
    else:
        keys_path = "./keys"
    
    success = generate_keys(keys_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
