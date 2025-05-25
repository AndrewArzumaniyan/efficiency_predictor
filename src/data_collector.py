import os
import shutil
import logging

logger = logging.getLogger(__name__)

def copy_and_remove(source, dest, output_file=None):
    """Копирует файл и удаляет оригинал"""
    try:
        if os.path.exists(source):
            shutil.copy2(source, dest)
            os.remove(source)
            message = f"Copied and removed {source} to {dest}"
            print(message)
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            return True
        else:
            error_msg = f"Исходный файл {source} не существует, копирование невозможно"
            print(error_msg)
            logging.error(error_msg)
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
            return False
    except Exception as e:
        error_msg = f"Error copying/removing {source} to {dest}: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
        return False
    
def cleanup_results_directories():
    print("Очистка директорий результатов, оставляем только info.json и лог-файлы...")
    
    results_dir = './results'
    if not os.path.exists(results_dir):
        print(f"Директория {results_dir} не существует, очистка не требуется")
        return
    
    # Перебираем все поддиректории
    for root, dirs, files in os.walk(results_dir):
        # Пропускаем корневую директорию
        if root == results_dir:
            continue
        
        # Проверяем каждый файл
        for file in files:
            file_path = os.path.join(root, file)
            
            # Сохраняем info.json и файлы с расширением .log
            if file == 'info.json' or file.endswith('_output.log') or file.endswith('.gcov'):
                continue  # Пропускаем эти файлы (не удаляем)
            else:
                try:
                    os.remove(file_path)
                    print(f"Удален файл: {file_path}")
                except Exception as e:
                    error_msg = f"Не удалось удалить файл {file_path}: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)

def find_directories_without_info_json():
    print("Поиск директорий без файла info.json...")
    
    results_dir = './results'
    if not os.path.exists(results_dir):
        print(f"Директория {results_dir} не существует")
        return []
    
    directories_without_info = []
    
    # Перебираем все поддиректории в results
    for root, dirs, files in os.walk(results_dir):
        # Пропускаем корневую директорию и директории верхнего уровня (1d, 2d, 3d, 4d)
        if root == results_dir or os.path.dirname(root) == results_dir:
            continue
        
        # Проверяем наличие info.json
        if 'info.json' not in files:
            directories_without_info.append(root)
            print(f"Директория без info.json: {root}")

    print(f"Всего найдено директорий без info.json: {len(directories_without_info)}")
    
    return directories_without_info