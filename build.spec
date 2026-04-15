# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata, collect_all

block_cipher = None

# --- 1. ЗБИРАЄМО STREAMLIT (Щоб працював інтерфейс) ---
streamlit_hidden_imports = collect_submodules('streamlit')
streamlit_metadata = copy_metadata('streamlit')
streamlit_datas = collect_data_files('streamlit')

# --- 2. ЗБИРАЄМО LLAMA_CPP (Щоб працював AI) ---
# Ця функція знаходить приховані .so / .dll файли
llama_datas, llama_binaries, llama_hiddenimports = collect_all('llama_cpp')

# --- 3. ОБ'ЄДНУЄМО ВСЕ РАЗОМ ---
hidden_imports = [
    'sqlite3',
    'pdfplumber',
    'docx',
    'graphviz',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes'
] + streamlit_hidden_imports + llama_hiddenimports

# Об'єднуємо файли даних
datas = [('app.py', '.')] + streamlit_metadata + streamlit_datas + llama_datas

# Виключаємо зайве
excluded_modules = [
    'torch', 'torchvision', 'torchaudio', 'pandas', 'matplotlib', 
    'tkinter', 'pyqt5', 'PyQt5', 'scipy', 'notebook', 'ipython',
    'lancedb', 'sentence_transformers'
]

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=llama_binaries, # <--- ВАЖЛИВО: Додаємо бінарники лами сюди
    datas=datas, 
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KOVCHEG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='KOVCHEG',
)
