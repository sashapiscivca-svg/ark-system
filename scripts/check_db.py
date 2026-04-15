#!/usr/bin/env python3
"""
check_db.py - Перевірка вмісту бази даних модуля
"""
import sys
import sqlite3
from pathlib import Path

def check_module(module_name):
    db_path = Path("modules") / module_name / "search.db"
    
    if not db_path.exists():
        print(f"❌ База даних не знайдена: {db_path}")
        return
    
    print(f"📊 Перевірка модуля '{module_name}'")
    print(f"📁 Шлях: {db_path}")
    print("=" * 60)
    
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    
    # Загальна статистика
    c.execute("SELECT COUNT(*) FROM docs")
    total_chunks = c.fetchone()[0]
    print(f"📚 Всього записів (chunks): {total_chunks:,}")
    
    # Унікальні файли
    c.execute("SELECT COUNT(DISTINCT filename) FROM docs")
    unique_files = c.fetchone()[0]
    print(f"📄 Унікальних файлів: {unique_files:,}")
    
    # Топ-10 файлів за кількістю chunks
    print("\n🔝 Топ-10 файлів за кількістю шматків:")
    c.execute("""
        SELECT filename, COUNT(*) as cnt 
        FROM docs 
        GROUP BY filename 
        ORDER BY cnt DESC 
        LIMIT 10
    """)
    for i, (filename, cnt) in enumerate(c.fetchall(), 1):
        print(f"  {i:2}. {filename[:50]:50} → {cnt:5,} шматків")
    
    # Перевірка векторів
    print("\n🔍 Перевірка векторів:")
    c.execute("SELECT id, LENGTH(embedding) FROM docs LIMIT 5")
    for doc_id, emb_size in c.fetchall():
        expected_size = 384 * 4  # 384 float32 = 1536 bytes
        status = "✅" if emb_size == expected_size else "❌"
        print(f"  {status} ID={doc_id}: {emb_size} bytes (очікувалось {expected_size})")
    
    # Граф знань
    c.execute("SELECT COUNT(*) FROM graph_edges")
    graph_edges = c.fetchone()[0]
    print(f"\n🕸️  Граф знань: {graph_edges:,} зв'язків")
    
    # Розмір бази
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 Розмір бази: {db_size_mb:.2f} MB")
    
    conn.close()
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Використання: python check_db.py <назва_модуля>")
        print("Приклад: python check_db.py Вікіпедія")
        sys.exit(1)
    
    check_module(sys.argv[1])
