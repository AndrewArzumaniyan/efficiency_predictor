#!/usr/bin/env python3
"""
Скрипт для дообучения моделей DVMH Performance Predictor
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging
from src.model_finetuner import DVMHModelFineTuner


def parse_strategies(strategies_str: str) -> list:
    """Парсинг списка стратегий из строки"""
    if not strategies_str:
        return ['full_model', 'last_layers', 'gradual_unfreezing']
    
    strategies = [s.strip() for s in strategies_str.split(',')]
    valid_strategies = ['full_model', 'last_layers', 'adapter', 'gradual_unfreezing']
    
    for strategy in strategies:
        if strategy not in valid_strategies:
            raise ValueError(f"Неизвестная стратегия: {strategy}. "
                           f"Доступные: {', '.join(valid_strategies)}")
    
    return strategies


def main():
    """Основная функция скрипта дообучения"""
    
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(
        description="Дообучение моделей DVMH Performance Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

1. Базовое дообучение:
   python scripts/finetune.py --base-model dvmh_mlp_model --new-data new_programs.csv

2. Дообучение с тестовыми данными:
   python scripts/finetune.py --base-model dvmh_attention_model --new-data new_data.csv --test-data test_data.csv

3. Тестирование конкретных стратегий:
   python scripts/finetune.py --base-model dvmh_mlp_model --new-data new_data.csv --strategies "full_model,last_layers"

4. Настройка параметров обучения:
   python scripts/finetune.py --base-model dvmh_mlp_model --new-data new_data.csv --lr 0.00005 --epochs 100 --batch-size 32

5. Просмотр дообученных моделей:
   python scripts/finetune.py --list-models

Стратегии дообучения:
  - full_model: Дообучение всей модели
  - last_layers: Дообучение только последних слоев
  - adapter: Добавление адаптерных слоев (экспериментально)
  - gradual_unfreezing: Постепенное размораживание слоев
        """
    )
    
    # Основные аргументы
    parser.add_argument(
        '--base-model', '-m',
        help='Имя базовой модели для дообучения (без расширения .pt)'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['mlp', 'attention'],
        default='mlp',
        help='Тип модели (по умолчанию: mlp)'
    )
    
    parser.add_argument(
        '--new-data', '-d',
        help='Путь к CSV файлу с новыми данными для дообучения'
    )
    
    parser.add_argument(
        '--test-data', '-t',
        help='Путь к CSV файлу с тестовыми данными (опционально)'
    )
    
    parser.add_argument(
        '--target',
        default='target_speedup',
        help='Название целевой переменной (по умолчанию: target_speedup)'
    )
    
    # Параметры дообучения
    parser.add_argument(
        '--strategies', '-s',
        help='Стратегии дообучения через запятую (по умолчанию: full_model,last_layers,gradual_unfreezing)'
    )
    
    parser.add_argument(
        '--adaptation',
        choices=['extend_preprocessor', 'refit_preprocessor', 'use_existing'],
        default='extend_preprocessor',
        help='Стратегия адаптации препроцессора (по умолчанию: extend_preprocessor)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Доля валидационной выборки (по умолчанию: 0.2)'
    )
    
    # Гиперпараметры
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.0001,
        help='Скорость обучения (по умолчанию: 0.0001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Количество эпох (по умолчанию: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Размер батча (по умолчанию: 64)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Терпение для ранней остановки (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Коэффициент регуляризации (по умолчанию: 0.01)'
    )
    
    # Утилитарные аргументы
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Показать список всех дообученных моделей'
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
    
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("DVMH Model Fine-Tuner - Дообучение моделей")
    logger.info("="*80)
    
    try:
        # Инициализируем fine-tuner
        finetuner = DVMHModelFineTuner(args.config)
        
        # Обработка команды списка моделей
        if args.list_models:
            models = finetuner.list_finetuned_models()
            
            if not models:
                print("❌ Дообученные модели не найдены")
                return 0
            
            print("📋 Список дообученных моделей:")
            print("-" * 100)
            print(f"{'Имя модели':<40} {'Базовая модель':<20} {'Стратегия':<15} {'R²':<8} {'Дата'}")
            print("-" * 100)
            
            for model in models:
                name = model['name'][:38] + ".." if len(model['name']) > 40 else model['name']
                base = model['base_model'][:18] + ".." if len(model['base_model']) > 20 else model['base_model']
                strategy = model['strategy'][:13] + ".." if len(model['strategy']) > 15 else model['strategy']
                r2 = f"{model['r2_score']:.4f}" if isinstance(model['r2_score'], (int, float)) else str(model['r2_score'])[:8]
                timestamp = model['timestamp']
                
                print(f"{name:<40} {base:<20} {strategy:<15} {r2:<8} {timestamp}")
            
            print("-" * 100)
            print(f"📁 Директория моделей: {finetuner.finetune_dir}")
            return 0
        
        # Проверяем основные аргументы
        if not args.base_model:
            print("❌ Не указана базовая модель")
            print("💡 Используйте --base-model для указания модели")
            return 1
        
        if not args.new_data:
            print("❌ Не указаны новые данные для дообучения")
            print("💡 Используйте --new-data для указания CSV файла")
            return 1
        
        # Проверяем существование файлов
        if not os.path.exists(args.new_data):
            print(f"❌ Файл с новыми данными не найден: {args.new_data}")
            return 1
        
        test_df = None
        if args.test_data:
            if not os.path.exists(args.test_data):
                print(f"❌ Файл с тестовыми данными не найден: {args.test_data}")
                return 1
            
            print(f"📊 Загрузка тестовых данных: {args.test_data}")
            test_df = pd.read_csv(args.test_data, low_memory=False)
            print(f"✅ Тестовые данные загружены: {test_df.shape}")
        
        # Загружаем новые данные
        print(f"📊 Загрузка новых данных: {args.new_data}")
        new_df = pd.read_csv(args.new_data, low_memory=False)
        print(f"✅ Новые данные загружены: {new_df.shape}")
        
        # Проверяем наличие целевой переменной
        if args.target not in new_df.columns:
            print(f"❌ Целевая переменная '{args.target}' не найдена в новых данных")
            print(f"💡 Доступные колонки: {list(new_df.columns)}")
            return 1
        
        if test_df is not None and args.target not in test_df.columns:
            print(f"❌ Целевая переменная '{args.target}' не найдена в тестовых данных")
            return 1
        
        # Парсим стратегии
        try:
            strategies = parse_strategies(args.strategies)
            print(f"🔧 Стратегии дообучения: {strategies}")
        except ValueError as e:
            print(f"❌ Ошибка в стратегиях: {str(e)}")
            return 1
        
        # Выводим параметры дообучения
        print(f"\n⚙️  Параметры дообучения:")
        print(f"   • Базовая модель: {args.base_model} ({args.model_type})")
        print(f"   • Адаптация препроцессора: {args.adaptation}")
        print(f"   • Валидационная выборка: {args.val_split * 100:.1f}%")
        print(f"   • Скорость обучения: {args.lr}")
        print(f"   • Эпохи: {args.epochs}")
        print(f"   • Размер батча: {args.batch_size}")
        print(f"   • Терпение: {args.patience}")
        print(f"   • Регуляризация: {args.weight_decay}")
        
        # Запускаем комплексное дообучение
        print(f"\n🚀 Начало дообучения...")
        
        results = finetuner.run_comprehensive_fine_tuning(
            new_df=new_df,
            base_model_name=args.base_model,
            model_type=args.model_type,
            target_column=args.target,
            test_df=test_df,
            fine_tune_strategies=strategies,
            validation_split=args.val_split,
            adaptation_strategy=args.adaptation,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience
        )
        
        # Выводим итоговые результаты
        print(f"\n" + "="*80)
        print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДООБУЧЕНИЯ")
        print("="*80)
        
        if results['best_strategy']:
            print(f"🏆 Лучшая стратегия: {results['best_strategy']}")
            
            best_metrics = results['best_metrics']
            print(f"📊 Метрики лучшей модели:")
            for metric, value in best_metrics.items():
                print(f"   • {metric.upper()}: {value:.4f}")
            
            # Сравнение с базовой моделью
            if results['base_model_metrics']:
                base_metrics = results['base_model_metrics']
                print(f"\n📈 Сравнение с базовой моделью:")
                
                for metric in best_metrics:
                    if metric in base_metrics:
                        base_val = base_metrics[metric]
                        new_val = best_metrics[metric]
                        
                        if metric in ['mse', 'mae', 'rmse']:
                            improvement = ((base_val - new_val) / base_val) * 100
                            symbol = "📉" if improvement > 0 else "📈"
                        else:  # r2
                            improvement = ((new_val - base_val) / base_val) * 100
                            symbol = "📈" if improvement > 0 else "📉"
                        
                        print(f"   {symbol} {metric.upper()}: {base_val:.4f} → {new_val:.4f} ({improvement:+.2f}%)")
        
        print(f"\n📁 Результаты сохранены в: {finetuner.finetune_dir}")
        
        # Рекомендации
        print(f"\n💡 Рекомендации:")
        if results['best_strategy']:
            best_model_name = results['strategies_results'][results['best_strategy']]['model_name']
            print(f"   • Для использования лучшей модели:")
            print(f"     python scripts/predict.py program.f --model {best_model_name} --grid '2,2' --threads 4 --time 1.5")
        
        print(f"   • Для просмотра всех дообученных моделей:")
        print(f"     python scripts/finetune.py --list-models")
        
        logger.info("Дообучение завершено успешно!")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Дообучение прервано пользователем")
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