#!/usr/bin/env python3
"""
Скрипт для предсказания эффективности DVMH программ
"""

import os
import sys
import argparse
import logging
import json

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging
from src.predictor import DVMHPerformancePredictor


def parse_grid(grid_str: str) -> list:
    """
    Парсинг строки с сеткой процессоров
    
    Args:
        grid_str: Строка в формате "1,2,3" или "1*2*3"
        
    Returns:
        list: Список размерностей сетки
    """
    try:
        # Поддерживаем разделители ',' и '*'
        if ',' in grid_str:
            grid = [int(x.strip()) for x in grid_str.split(',')]
        elif '*' in grid_str:
            grid = [int(x.strip()) for x in grid_str.split('*')]
        else:
            # Одно число
            grid = [int(grid_str.strip())]
        
        # Валидация
        if not all(g > 0 for g in grid):
            raise ValueError("Все размерности сетки должны быть положительными")
        
        return grid
    except Exception as e:
        raise ValueError(f"Некорректный формат сетки '{grid_str}': {str(e)}")


def validate_input_file(file_path: str) -> str:
    """
    Валидация и определение типа входного файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        str: Тип файла ('fortran' или 'json')
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in ['.f', '.f90', '.for', '.fortran']:
        return 'fortran'
    elif file_ext in ['.json']:
        return 'json'
    else:
        raise ValueError(f"Неподдерживаемый тип файла: {file_ext}. "
                        f"Поддерживаются: .f, .f90, .for, .fortran, .json")


def list_available_models(predictor: DVMHPerformancePredictor) -> None:
    """Вывод списка доступных моделей"""
    models = predictor.list_available_models()
    
    if not models:
        print("❌ Не найдено доступных моделей")
        print("💡 Сначала обучите модель с помощью команды 'python main.py train'")
        return
    
    print("📋 Доступные модели:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    print(f"\n📁 Директория моделей: {predictor.models_dir}")


def interactive_model_selection(predictor: DVMHPerformancePredictor) -> tuple:
    """Интерактивный выбор модели"""
    models = predictor.list_available_models()
    
    if not models:
        print("❌ Не найдено доступных моделей для интерактивного выбора")
        return None, None
    
    print("\n🔍 Выберите модель для предсказания:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input(f"\nВведите номер модели (1-{len(models)}): ").strip()
            model_idx = int(choice) - 1
            
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                
                # Определяем тип модели
                model_type = 'attention' if 'attention' in selected_model.lower() else 'mlp'
                
                print(f"✅ Выбрана модель: {selected_model} (тип: {model_type})")
                return selected_model, model_type
            else:
                print(f"❌ Неверный выбор. Введите число от 1 до {len(models)}")
                
        except ValueError:
            print("❌ Введите корректное число")
        except KeyboardInterrupt:
            print("\n❌ Выбор модели отменен")
            return None, None


def extract_speedup_from_result(result) -> float:
    """
    Извлечение значения ускорения из результата предсказания
    
    Args:
        result: Результат от predict_from_* методов (может быть float или tuple)
        
    Returns:
        float: Значение ускорения
    """
    if isinstance(result, tuple):
        # Для attention модели возвращается (speedup, attention_weights)
        return result[0]
    else:
        # Для обычной модели возвращается просто speedup
        return result


def calculate_grid_size(grid: list) -> int:
    """
    Вычисление общего размера сетки процессоров
    
    Args:
        grid: Список размерностей сетки
        
    Returns:
        int: Общее количество процессоров
    """
    result = 1
    for dim in grid:
        result *= dim
    return result


def main():
    """Основная функция скрипта"""
    
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(
        description="Предсказание эффективности DVMH программ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Предсказание из Fortran файла:
   python scripts/predict.py program.f --grid "2,2" --threads 4 --time 1.5

2. Предсказание из JSON файла статистики:
   python scripts/predict.py info.json --grid "1*2*3" --threads 8 --time 2.1

3. Список доступных моделей:
   python scripts/predict.py --list-models

4. Интерактивный режим:
   python scripts/predict.py program.f --grid "2,2" --threads 4 --time 1.5 --interactive

5. Использование конкретной модели:
   python scripts/predict.py program.f --grid "2,2" --threads 4 --time 1.5 --model dvmh_attention_model --model-type attention

Форматы сетки процессоров:
  - "2,2" или "2*2" для двумерной сетки 2x2
  - "1,2,3" или "1*2*3" для трехмерной сетки 1x2x3
  - "4" для одномерной сетки из 4 процессоров
        """
    )
    
    # Основные аргументы
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Путь к Fortran файлу (.f, .f90) или JSON файлу статистики (.json)'
    )
    
    parser.add_argument(
        '--grid', '-g',
        help='Сетка процессоров в формате "x,y,z" или "x*y*z"'
    )
    
    parser.add_argument(
        '--threads', '-t',
        type=int,
        help='Количество нитей'
    )
    
    parser.add_argument(
        '--time',
        type=float,
        help='Время параллельного выполнения (в секундах)'
    )
    
    # Настройки модели
    parser.add_argument(
        '--model', '-m',
        help='Имя модели для использования (без расширения .pt)'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['mlp', 'attention'],
        default='mlp',
        help='Тип модели (по умолчанию: mlp)'
    )
    
    # Утилитарные аргументы
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Показать список доступных моделей'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Интерактивный выбор модели'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/data_config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: config/data_config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод (уровень логирования DEBUG)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['simple', 'json', 'detailed'],
        default='simple',
        help='Формат вывода результата (по умолчанию: simple)'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("DVMH Performance Predictor - Предсказание эффективности")
    logger.info("="*60)
    
    try:
        # Инициализируем предиктор
        predictor = DVMHPerformancePredictor(args.config)
        
        # Обработка команды списка моделей
        if args.list_models:
            list_available_models(predictor)
            return 0
        
        # Проверяем основные аргументы
        if not args.input_file:
            if args.interactive:
                print("❌ Для интерактивного режима необходимо указать входной файл")
            else:
                print("❌ Не указан входной файл")
                print("💡 Используйте --help для просмотра справки")
            return 1
        
        if not all([args.grid, args.threads is not None, args.time is not None]):
            print("❌ Необходимо указать все параметры: --grid, --threads, --time")
            print("💡 Используйте --help для просмотра примеров")
            return 1
        
        # Валидация входного файла
        try:
            file_type = validate_input_file(args.input_file)
            print(f"📄 Тип файла: {file_type}")
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Ошибка входного файла: {str(e)}")
            return 1
        
        # Парсинг сетки процессоров
        try:
            grid = parse_grid(args.grid)
            print(f"🔧 Сетка процессоров: {grid}")
        except ValueError as e:
            print(f"❌ Ошибка в параметре сетки: {str(e)}")
            return 1
        
        # Валидация остальных параметров
        if args.threads <= 0:
            print("❌ Количество нитей должно быть положительным числом")
            return 1
        
        if args.time <= 0:
            print("❌ Время выполнения должно быть положительным числом")
            return 1
        
        print(f"⚙️  Параметры запуска:")
        print(f"   • Нити: {args.threads}")
        print(f"   • Время: {args.time} сек")
        
        # Выбор модели
        if args.interactive:
            model_name, model_type = interactive_model_selection(predictor)
            if model_name is None:
                return 1
        else:
            if args.model:
                model_name = args.model
                model_type = args.model_type
            else:
                # Автоматический выбор первой доступной модели
                available_models = predictor.list_available_models()
                if not available_models:
                    print("❌ Не найдено доступных моделей")
                    print("💡 Сначала обучите модель с помощью команды 'python main.py train'")
                    return 1
                
                model_name = available_models[0]
                model_type = 'attention' if 'attention' in model_name.lower() else 'mlp'
                print(f"🤖 Автоматически выбрана модель: {model_name} (тип: {model_type})")
        
        # Загрузка модели
        print(f"⏳ Загрузка модели {model_name}...")
        if not predictor.load_model(model_name, model_type):
            print(f"❌ Не удалось загрузить модель {model_name}")
            return 1
        
        # Получение информации о модели
        model_info = predictor.get_model_info(model_name)
        if model_info and args.output_format in ['detailed', 'json']:
            print(f"📊 Информация о модели:")
            print(f"   • Тип: {model_info['model_type']}")
            print(f"   • Размерность входа: {model_info['input_dim']}")
            print(f"   • Архитектура: {model_info['hidden_dims']}")
            print(f"   • Количество признаков: {model_info['feature_count']}")
            if 'metrics' in model_info and model_info['metrics']:
                metrics = model_info['metrics']
                if 'r2' in metrics:
                    print(f"   • R² Score: {metrics['r2']:.4f}")
                if 'mae' in metrics:
                    print(f"   • MAE: {metrics['mae']:.4f}")
        
        # Выполнение предсказания
        print(f"🔮 Выполнение предсказания...")
        
        try:
            if file_type == 'fortran':
                result = predictor.predict_from_fortran(
                    args.input_file, grid, args.threads, args.time
                )
            else:  # json
                result = predictor.predict_from_json(
                    args.input_file, grid, args.threads, args.time
                )
        except Exception as e:
            print(f"❌ Ошибка при выполнении предсказания: {str(e)}")
            if args.verbose:
                logger.exception("Детали ошибки предсказания:")
            return 1
        
        # Проверка результата
        if result is None:
            print("❌ Не удалось выполнить предсказание")
            return 1
        
        # Извлекаем ускорение из результата
        speedup = extract_speedup_from_result(result)
        
        # Форматированный вывод результата
        if args.output_format == 'simple':
            print(f"\n🎯 Предсказанное ускорение: {speedup:.4f}x")
            
        elif args.output_format == 'json':
            result_data = {
                'input_file': args.input_file,
                'file_type': file_type,
                'grid': grid,
                'threads': args.threads,
                'parallel_time': args.time,
                'predicted_speedup': float(speedup),
                'model_name': model_name,
                'model_type': model_type
            }
            print(json.dumps(result_data, indent=2, ensure_ascii=False))
            
        elif args.output_format == 'detailed':
            print(f"\n" + "="*50)
            print(f"📋 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
            print(f"="*50)
            print(f"📄 Входной файл: {args.input_file}")
            print(f"📂 Тип файла: {file_type}")
            print(f"🔧 Сетка процессоров: {grid}")
            print(f"⚙️  Количество нитей: {args.threads}")
            print(f"⏱️  Время выполнения: {args.time} сек")
            print(f"🤖 Модель: {model_name} ({model_type})")
            print(f"🎯 Предсказанное ускорение: {speedup:.4f}x")
            
            # Дополнительная аналитика
            if args.time > 0:
                estimated_sequential_time = args.time * speedup
                grid_size = calculate_grid_size(grid)
                total_processors = args.threads * grid_size
                efficiency = speedup / total_processors if total_processors > 0 else 0
                
                print(f"📈 Предполагаемое последовательное время: {estimated_sequential_time:.4f} сек")
                print(f"📊 Эффективность параллелизации: {efficiency:.4f}")
                print(f"🖥️  Общее количество процессоров: {total_processors}")
            
            print(f"="*50)
        
        # Дополнительные рекомендации
        grid_size = calculate_grid_size(grid)
        total_processors = args.threads * grid_size
        
        if speedup < 1.0:
            print(f"⚠️  Предупреждение: Предсказанное ускорение меньше 1.0")
            print(f"   Это может указывать на неэффективную параллелизацию")
        elif speedup > total_processors * 1.5:  # Разрешаем разумное суперлинейное ускорение
            print(f"⚠️  Предупреждение: Очень высокое ускорение ({speedup:.2f}x > {total_processors * 1.5:.1f}x)")
            print(f"   Проверьте корректность входных данных")
        elif speedup > total_processors:
            print(f"ℹ️  Информация: Суперлинейное ускорение ({speedup:.2f}x > {total_processors}x)")
            print(f"   Это возможно благодаря эффектам кэша и лучшей утилизации памяти")
        else:
            efficiency = speedup / total_processors
            if efficiency > 0.8:
                print(f"✅ Отличная эффективность параллелизации ({efficiency:.1%})")
            elif efficiency > 0.6:
                print(f"✅ Хорошая эффективность параллелизации ({efficiency:.1%})")
            else:
                print(f"⚠️  Низкая эффективность параллелизации ({efficiency:.1%})")
        
        logger.info("Предсказание завершено успешно")
        logger.info("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Выполнение прервано пользователем")
        return 130
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        if args.verbose:
            logger.exception("Детали ошибки:")
        print(f"❌ Произошла ошибка: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)