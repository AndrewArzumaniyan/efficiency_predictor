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


def main():
    """Основная функция скрипта предсказания"""
    
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(
        description="Предсказание эффективности DVMH программ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  
  Предсказание для одной конфигурации:
  python scripts/predict.py single --program program.json --grid 2 2 --threads 4 --model dvmh_mlp_model
  
  Поиск оптимальной конфигурации:
  python scripts/predict.py optimize --program program.json --max-processors 16 --model dvmh_attention_model
  
  Пакетное предсказание:
  python scripts/predict.py batch --input batch_input.json --output results.json
        """
    )
    
    # Основные аргументы
    parser.add_argument(
        'mode',
        choices=['single', 'optimize', 'batch', 'info'],
        help='Режим работы: single - одна конфигурация, optimize - поиск оптимальной, batch - пакетное предсказание, info - информация о модели'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/data_config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: config/data_config.yaml)'
    )
    
    parser.add_argument(
        '--model', '-m',
        help='Имя модели для загрузки (без расширения)'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['mlp', 'attention'],
        default='mlp',
        help='Тип модели (по умолчанию: mlp)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод (уровень логирования DEBUG)'
    )
    
    # Аргументы для режима single
    single_group = parser.add_argument_group('single', 'Аргументы для одиночного предсказания')
    single_group.add_argument(
        '--program',
        help='Путь к файлу с данными программы (JSON)'
    )
    single_group.add_argument(
        '--grid',
        nargs='+',
        type=int,
        help='Размерности сетки процессоров (например: 2 2 для 2x2)'
    )
    single_group.add_argument(
        '--threads',
        type=int,
        help='Количество нитей'
    )
    single_group.add_argument(
        '--dvmh-processors',
        type=int,
        default=1,
        help='Количество DVMH процессоров (по умолчанию: 1)'
    )
    
    # Аргументы для режима optimize
    optimize_group = parser.add_argument_group('optimize', 'Аргументы для оптимизации')
    optimize_group.add_argument(
        '--max-processors',
        type=int,
        default=16,
        help='Максимальное количество процессоров для поиска (по умолчанию: 16)'
    )
    
    # Аргументы для режима batch
    batch_group = parser.add_argument_group('batch', 'Аргументы для пакетного предсказания')
    batch_group.add_argument(
        '--input', '-i',
        help='Входной файл с данными для пакетного предсказания'
    )
    batch_group.add_argument(
        '--output', '-o',
        help='Выходной файл для результатов'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Запуск предсказания DVMH Performance Predictor")
    logger.info(f"Режим: {args.mode}")
    if args.model:
        logger.info(f"Модель: {args.model} ({args.model_type})")
    logger.info("="*60)
    
    try:
        # Инициализируем предиктор
        predictor = DVMHPerformancePredictor(args.config)
        
        # Загружаем модель, если указана
        if args.model:
            if not predictor.load_model(args.model, args.model_type):
                logger.error("Не удалось загрузить модель")
                return 1
            
            # Пытаемся загрузить скейлер
            scaler_path = os.path.join(os.path.dirname(predictor.models_dir), 
                                     'processed', 'scaler.pkl')
            if os.path.exists(scaler_path):
                predictor.load_scaler(scaler_path)
            else:
                logger.warning("Скейлер не найден, используем данные без нормализации")
        
        # Выполняем действие в зависимости от режима
        if args.mode == 'single':
            exit_code = mode_single(args, predictor, logger)
        elif args.mode == 'optimize':
            exit_code = mode_optimize(args, predictor, logger)
        elif args.mode == 'batch':
            exit_code = mode_batch(args, predictor, logger)
        elif args.mode == 'info':
            exit_code = mode_info(args, predictor, logger)
        else:
            logger.error(f"Неизвестный режим: {args.mode}")
            exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Предсказание прервано пользователем")
        return 130
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        if args.verbose:
            logger.exception("Детали ошибки:")
        return 1


def mode_single(args, predictor, logger):
    """Режим одиночного предсказания"""
    logger.info("=== РЕЖИМ: Одиночное предсказание ===")
    
    # Проверяем аргументы
    if not all([args.program, args.grid, args.threads]):
        logger.error("Для режима single необходимы аргументы: --program, --grid, --threads")
        return 1
    
    if not os.path.exists(args.program):
        logger.error(f"Файл программы не найден: {args.program}")
        return 1
    
    try:
        # Загружаем данные программы
        program_data = predictor.load_program_data(args.program)
        
        # Делаем предсказание
        result = predictor.predict_single_configuration(
            program_data=program_data,
            grid=args.grid,
            threads=args.threads,
            model_name=args.model,
            dvmh_processors=args.dvmh_processors
        )
        
        # Выводим результат
        if 'error' in result:
            logger.error(f"Ошибка предсказания: {result['error']}")
            return 1
        
        logger.info("Результат предсказания:")
        logger.info(f"  Конфигурация: grid={result['grid']}, threads={result['threads']}")
        logger.info(f"  Предсказанное ускорение: {result['predicted_speedup']:.3f}")
        logger.info(f"  Общее количество процессоров: {result['total_processors']}")
        logger.info(f"  Модель: {result['model_used']} ({result['model_type']})")
        
        if 'top_important_features' in result:
            logger.info("  Наиболее важные признаки:")
            for feature in result['top_important_features']:
                logger.info(f"    {feature['feature']}: {feature['feature_value']:.3f} "
                          f"(вес: {feature['attention_weight']:.3f})")
        
        # Сохраняем результат
        output_file = f"single_prediction_{args.grid}_{args.threads}.json"
        predictor.save_predictions(result, output_file)
        
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка в режиме single: {str(e)}")
        return 1


def mode_optimize(args, predictor, logger):
    """Режим поиска оптимальной конфигурации"""
    logger.info("=== РЕЖИМ: Поиск оптимальной конфигурации ===")
    
    if not args.program:
        logger.error("Для режима optimize необходим аргумент: --program")
        return 1
    
    if not os.path.exists(args.program):
        logger.error(f"Файл программы не найден: {args.program}")
        return 1
    
    try:
        # Загружаем данные программы
        program_data = predictor.load_program_data(args.program)
        
        # Ищем оптимальную конфигурацию
        optimization_result = predictor.find_optimal_configuration(
            program_data=program_data,
            max_processors=args.max_processors,
            model_name=args.model,
            dvmh_processors=args.dvmh_processors
        )
        
        # Выводим результат
        if 'error' in optimization_result:
            logger.error(f"Ошибка оптимизации: {optimization_result['error']}")
            return 1
        
        best_config = optimization_result['best_configuration']
        
        logger.info("Результат оптимизации:")
        logger.info(f"  Лучшая конфигурация: grid={best_config['grid']}, threads={best_config['threads']}")
        logger.info(f"  Максимальное ускорение: {best_config['predicted_speedup']:.3f}")
        logger.info(f"  Проверено конфигураций: {optimization_result['configurations_tested']}")
        logger.info(f"  Валидных предсказаний: {optimization_result['valid_predictions']}")
        
        speedup_range = optimization_result['speedup_range']
        logger.info(f"  Диапазон ускорений: {speedup_range['min']:.3f} - {speedup_range['max']:.3f}")
        logger.info(f"  Среднее ускорение: {speedup_range['mean']:.3f}")
        
        # Сохраняем результат
        output_file = f"optimization_result_max{args.max_processors}.json"
        predictor.save_predictions(optimization_result, output_file)
        
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка в режиме optimize: {str(e)}")
        return 1


def mode_batch(args, predictor, logger):
    """Режим пакетного предсказания"""
    logger.info("=== РЕЖИМ: Пакетное предсказание ===")
    
    if not args.input:
        logger.error("Для режима batch необходим аргумент: --input")
        return 1
    
    if not os.path.exists(args.input):
        logger.error(f"Входной файл не найден: {args.input}")
        return 1
    
    try:
        # Выполняем пакетное предсказание
        output_path = predictor.batch_predict_from_file(
            input_file=args.input,
            output_file=args.output,
            model_name=args.model
        )
        
        logger.info(f"Пакетное предсказание завершено: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка в режиме batch: {str(e)}")
        return 1


def mode_info(args, predictor, logger):
    """Режим получения информации о модели"""
    logger.info("=== РЕЖИМ: Информация о модели ===")
    
    try:
        info = predictor.get_model_info(args.model)
        
        if 'error' in info:
            logger.error(info['error'])
            return 1
        
        logger.info("Информация о модели:")
        logger.info(f"  Имя: {info['model_name']}")
        logger.info(f"  Тип: {info['model_type']}")
        logger.info(f"  Размерность входа: {info['input_dim']}")
        logger.info(f"  Скрытые слои: {info['hidden_dims']}")
        logger.info(f"  Количество признаков: {info['feature_count']}")
        
        if info['metrics']:
            logger.info("  Метрики модели:")
            for metric, value in info['metrics'].items():
                logger.info(f"    {metric}: {value:.4f}")
        
        if info['test_programs']:
            logger.info(f"  Тестовые программы: {info['test_programs']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка в режиме info: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)