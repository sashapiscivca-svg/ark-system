import sys
import os
from streamlit.web import cli as stcli
from multiprocessing import freeze_support

if __name__ == '__main__':
    # 1. КРИТИЧНО: Зупиняє нескінченне розмноження процесів (рятує RAM)
    freeze_support()

    # 2. Визначаємо реальні шляхи
    if getattr(sys, 'frozen', False):
        # Якщо запущено як скомпільований файл (на флешці)
        app_root = os.path.dirname(sys.executable)
        # Шлях до app.py всередині тимчасової папки PyInstaller
        script_path = os.path.join(sys._MEIPASS, 'app.py')
    else:
        # Для тестів
        app_root = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(app_root, 'app.py')

    # 3. Передаємо координати флешки в app.py через пам'ять процесу
    os.environ['KOVCHEG_ROOT'] = app_root
    
    # 4. Примусово переходимо в папку на флешці
    os.chdir(app_root)
    
    print(f"📍 REAL ROOT: {app_root}")
    print(f"📜 SCRIPT: {script_path}")

    # 5. Налаштування запуску Streamlit
    sys.argv = [
        "streamlit", 
        "run", 
        script_path,
        "--global.developmentMode=false", 
        "--server.headless=true", 
        "--server.fileWatcherType=none", # Вимикаємо стеження за файлами (важливо для Linux USB)
        "--browser.gatherUsageStats=false",
        "--theme.base=dark"
    ]
    
    sys.exit(stcli.main())
