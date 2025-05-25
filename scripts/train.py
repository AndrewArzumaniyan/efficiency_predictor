#!/usr/bin/env python3
"""
Скрипт для обучения моделей предсказания эффективности DVMH программ
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging
from src.model_trainer import DVMHModelTrainer


def main():
    """Основная функция скрипта обучения"""
    
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(
        description="Обучение моделей предсказания эффективности DVMH программ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/train.py --dataset data/processed/feature_dataset.csv
  python scripts/train.py --dataset dataset.csv --target target_speedup --epochs 150
  python scripts/train.py --dataset dataset.csv --test-programs 5 --hidden-dims 512 256 128
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Путь к датасету для обучения (CSV файл)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/data_config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: config/data_config.yaml)'
    )
    
    parser.add_argument(
        '--target', '-t',
        default='target_speedup',
        help='Название целевой переменной (по умолчанию: target_speedup)'
    )
    
    parser.add_argument(
        '--test-programs',
        type=int,
        default=3,
        help='Количество программ для тестирования (по умолчанию: 3)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Доля валидационной выборки (по умолчанию: 0.2)'
    )
    
    parser.add_argument(
        '--hidden-dims',
        nargs='+',
        type=int,
        default=[256, 128, 64],
        help='Размеры скрытых слоев (по умолчанию: 256 128 64)'
    )
    
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=64,
        help='Размерность слоя внимания (по умолчанию: 64)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Количество эпох обучения (по умолчанию: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Размер батча (по умолчанию: 128)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Скорость обучения (по умолчанию: 0.001)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Терпение для ранней остановки (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод (уровень логирования DEBUG)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Случайное семя для воспроизводимости (по умолчанию: 42)'
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Запуск обучения моделей DVMH Performance Predictor")
    logger.info(f"Датасет: {args.dataset}")
    logger.info(f"Конфигурация: {args.config}")
    logger.info("="*60)
    
    try:
        # Проверяем существование файлов
        if not os.path.exists(args.dataset):
            logger.error(f"Датасет не найден: {args.dataset}")
            return 1
        
        if not os.path.exists(args.config):
            logger.warning(f"Конфигурационный файл не найден: {args.config}")
            logger.info("Будут использованы настройки по умолчанию")
        
        # Устанавливаем семя для воспроизводимости
        import random
        import numpy as np
        import torch
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        
        logger.info(f"Установлено случайное семя: {args.seed}")
        
        # Загружаем датасет
        logger.info("Загрузка датасета...")
        try:
            df = pd.read_csv(args.dataset, low_memory=False)
            logger.info(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {str(e)}")
            return 1
        
        # Проверяем наличие целевой переменной
        if args.target not in df.columns:
            logger.error(f"Целевая переменная '{args.target}' не найдена в датасете")
            logger.info(f"Доступные столбцы: {list(df.columns)}")
            return 1
        
        # Проверяем наличие программ
        if 'program_name' not in df.columns:
            logger.error("Столбец 'program_name' не найден в датасете")
            return 1
        
        unique_programs = df['program_name'].nunique()
        if unique_programs < args.test_programs:
            logger.error(f"Недостаточно программ для тестирования. "
                        f"Доступно: {unique_programs}, требуется: {args.test_programs}")
            return 1
        
        # Инициализируем тренер
        trainer = DVMHModelTrainer(args.config)
        
        # Запускаем обучение
        logger.info("Начало обучения моделей...")
        
        results = trainer.run_full_training(
            df=df,
            target_column=args.target,
            test_programs_count=args.test_programs,
            val_size=args.val_size,
            hidden_dims=args.hidden_dims,
            attention_dim=args.attention_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        
        # Выводим итоговые результаты
        logger.info("="*60)
        logger.info("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        logger.info("="*60)
        
        logger.info("Размеры данных:")
        for key, value in results['data_info'].items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\nТестовые программы: {results['test_programs']}")
        
        logger.info("\nМетрики MLP модели:")
        for metric, value in results['mlp_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nМетрики Attention модели:")
        for metric, value in results['attention_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Сравнение моделей
        mlp_r2 = results['mlp_metrics']['r2']
        attention_r2 = results['attention_metrics']['r2']
        
        if attention_r2 > mlp_r2:
            improvement = ((attention_r2 - mlp_r2) / mlp_r2) * 100
            logger.info(f"\n🎯 Модель с вниманием показала лучший результат!")
            logger.info(f"   Улучшение R²: {improvement:.2f}%")
        elif mlp_r2 > attention_r2:
            logger.info(f"\n🎯 MLP модель показала лучший результат!")
        else:
            logger.info(f"\n🎯 Обе модели показали одинаковые результаты")
        
        logger.info(f"\nМодели сохранены в: {trainer.models_dir}")
        logger.info(f"Результаты и графики сохранены в: {os.path.join(trainer.models_dir, 'results')}")
        
        logger.info("="*60)
        logger.info("Обучение завершено успешно!")
        logger.info("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Обучение прервано пользователем")
        return 130
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        if args.verbose:
            logger.exception("Детали ошибки:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)