import os
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from .utils import run_command

logger = logging.getLogger(__name__)


class DVMHStatsAggregator:
    """Класс для агрегации статистики покрытия и результатов параллельных запусков"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация агрегатора статистики
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)
        
        # Извлекаем пути из конфига
        aggregation_config = self.config.get('data_aggregation', {})
        self.cover_path = aggregation_config.get('cover_directory', "./data/raw")
        self.results_path = aggregation_config.get('results_directory', "./data/parallel_results")
        self.output_path = aggregation_config.get('output_directory', "./data/processed")
        self.dimensions = self.config['data_collection']['dimensions']
        
        # Регулярные выражения для парсинга файлов статистики
        self.processor_pattern = r'Processor system=(.+)'
        self.threads_pattern = r'Threads amount\s+(\d+)'
        self.time_pattern = r'Total time\s+([\d.]+)'
        
        self.logger.info("DVMHStatsAggregator инициализирован")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"Конфигурационный файл не найден: {config_path}")
            # Возвращаем конфигурацию по умолчанию
            return {
                'data_collection': {
                    'dimensions': ['1d', '2d', '3d', '4d']
                },
                'data_aggregation': {
                    'cover_directory': './data/raw',
                    'results_directory': './data/parallel_results', 
                    'output_directory': './data/processed'
                }
            }
    
    def aggregate_all_statistics(self) -> Dict[str, Dict]:
        """
        Агрегация всей статистики: покрытие + результаты запусков
        
        Returns:
            Dict: Словарь с агрегированными данными по всем программам
        """
        self.logger.info("Начало агрегации статистики")
        
        # 1. Загружаем данные покрытия
        coverage_programs = self._load_coverage_data()
        if not coverage_programs:
            self.logger.error("Не удалось загрузить данные покрытия")
            return {}
        
        # 2. Обрабатываем каждую программу
        processed_programs = {}
        deleted_programs = []
        
        for program_key, program_data in coverage_programs.items():
            try:
                processed_program = self._process_single_program(program_key, program_data)
                if processed_program:
                    # Используем только имя программы без размерности как ключ
                    program_name = program_key.split('/')[1]
                    processed_programs[program_name] = processed_program
                else:
                    deleted_programs.append(program_key)
            except Exception as e:
                self.logger.error(f"Ошибка при обработке программы {program_key}: {str(e)}")
                deleted_programs.append(program_key)
        
        # 3. Логируем результаты
        self.logger.info(f"Успешно обработано программ: {len(processed_programs)}")
        self.logger.info(f"Программы с ошибками: {len(deleted_programs)}")
        
        if deleted_programs:
            self.logger.warning(f"Проблемные программы: {deleted_programs}")
        
        return processed_programs
    
    def _load_coverage_data(self) -> Dict[str, Dict]:
        """Загрузка данных покрытия из файлов info.json"""
        self.logger.info(f"Загрузка данных покрытия из {self.cover_path}")
        
        programs = {}
        
        for dim in self.dimensions:
            cover_dim_path = os.path.join(self.cover_path, dim)
            
            if not os.path.exists(cover_dim_path):
                self.logger.warning(f"Директория размерности не найдена: {cover_dim_path}")
                continue
            
            try:
                program_dirs = os.listdir(cover_dim_path)
            except OSError as e:
                self.logger.error(f"Ошибка при чтении директории {cover_dim_path}: {str(e)}")
                continue
            
            for program_dir in program_dirs:
                program_info_path = os.path.join(cover_dim_path, program_dir, 'info.json')
                
                try:
                    with open(program_info_path, 'r', encoding='utf-8') as f:
                        program_data = json.load(f)
                        programs[f"{dim}/{program_dir}"] = program_data
                        self.logger.debug(f"Загружены данные покрытия: {dim}/{program_dir}")
                except FileNotFoundError:
                    self.logger.warning(f"Файл info.json не найден: {program_info_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Ошибка при парсинге JSON {program_info_path}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Неожиданная ошибка при загрузке {program_info_path}: {str(e)}")
        
        self.logger.info(f"Загружено программ с данными покрытия: {len(programs)}")
        return programs
    
    def _process_single_program(self, program_key: str, program_data: Dict) -> Optional[Dict]:
        """
        Обработка одной программы: объединение покрытия и результатов запусков
        
        Args:
            program_key: Ключ программы в формате "dim/program_name"
            program_data: Данные покрытия программы
            
        Returns:
            Dict или None: Обработанные данные программы
        """
        self.logger.debug(f"Обработка программы: {program_key}")
        
        # Извлекаем информацию из данных покрытия
        if len(program_data) < 2:
            self.logger.error(f"Некорректный формат данных покрытия для {program_key}")
            return None
        
        cluster_info = program_data[0].get('cluster_info', {})
        program_info = program_data[1].get('program_info', {})
        
        # Удаляем launch_grid если есть (как в оригинальном коде)
        if 'launch_grid' in program_info:
            del program_info['launch_grid']
        
        # Создаем структуру обработанной программы
        processed_program = {
            'cluster_info': cluster_info,
            'program_info': program_info.copy()
        }
        processed_program['program_info']['launches'] = []
        
        # Загружаем результаты запусков
        results_path = os.path.join(self.results_path, program_key)
        self.logger.debug(f"Поиск результатов запусков в: {results_path}")
        
        if not os.path.exists(results_path):
            self.logger.warning(f"Директория с результатами не найдена: {results_path}")
            return None
        
        try:
            runs_dirs = os.listdir(results_path)
        except OSError as e:
            self.logger.error(f"Ошибка при чтении директории результатов {results_path}: {str(e)}")
            return None
        
        # Обрабатываем каждый запуск
        sequential_time_found = False
        
        for run_dir in runs_dirs:
            run_stat_file = os.path.join(results_path, run_dir, 'stat.txt')
            
            try:
                run_info = self._parse_run_statistics(run_stat_file)
                if run_info:
                    if run_info['threads'] == 1:
                        # Последовательное выполнение
                        processed_program['program_info']['sequential_execution_time_sec'] = run_info['total_time']
                        sequential_time_found = True
                        self.logger.debug(f"Найдено последовательное время: {run_info['total_time']}с")
                    else:
                        # Параллельное выполнение
                        launch_data = {
                            'grid': run_info['grid'],
                            'threads': run_info['threads'],
                            'total_time': run_info['total_time']
                        }
                        processed_program['program_info']['launches'].append(launch_data)
                        self.logger.debug(f"Добавлен запуск: grid={run_info['grid']}, threads={run_info['threads']}")
                        
            except Exception as e:
                self.logger.warning(f"Ошибка при обработке запуска {run_stat_file}: {str(e)}")
                continue
        
        # Проверяем, что у нас есть необходимые данные
        if not sequential_time_found:
            self.logger.warning(f"Не найдено время последовательного выполнения для {program_key}")
        
        if not processed_program['program_info']['launches']:
            self.logger.warning(f"Не найдено параллельных запусков для {program_key}")
            return None
        
        self.logger.debug(f"Программа {program_key} обработана успешно. Запусков: {len(processed_program['program_info']['launches'])}")
        return processed_program
    
    def _parse_run_statistics(self, stat_file_path: str) -> Optional[Dict]:
        """
        Парсинг файла статистики одного запуска
        
        Args:
            stat_file_path: Путь к файлу stat.txt
            
        Returns:
            Dict или None: Информация о запуске
        """
        if not os.path.exists(stat_file_path):
            self.logger.debug(f"Файл статистики не найден: {stat_file_path}")
            return None
        
        try:
            with open(stat_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Извлекаем данные с помощью регулярных выражений
            processor_match = re.search(self.processor_pattern, content)
            threads_match = re.search(self.threads_pattern, content)
            time_match = re.search(self.time_pattern, content)
            
            if not all([processor_match, threads_match, time_match]):
                self.logger.warning(f"Не все данные найдены в файле: {stat_file_path}")
                return None
            
            processor_system = processor_match.group(1)
            threads_amount = int(threads_match.group(1))
            total_time = float(time_match.group(1))
            
            # Парсим grid из processor_system
            grid = list(map(int, processor_system.split('*')))
            
            return {
                'grid': grid,
                'threads': threads_amount,
                'total_time': total_time
            }
            
        except (IOError, ValueError, AttributeError) as e:
            self.logger.error(f"Ошибка при парсинге файла {stat_file_path}: {str(e)}")
            return None
    
    def save_aggregated_data(self, data: Dict[str, Dict], filename: str = "aggregated_data.json") -> bool:
        """
        Сохранение агрегированных данных в файл
        
        Args:
            data: Агрегированные данные
            filename: Имя файла для сохранения
            
        Returns:
            bool: True если сохранение прошло успешно
        """
        # Создаем выходную директорию
        os.makedirs(self.output_path, exist_ok=True)
        
        output_file = os.path.join(self.output_path, filename)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Агрегированные данные сохранены в: {output_file}")
            self.logger.info(f"Всего программ в файле: {len(data)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении данных в {output_file}: {str(e)}")
            return False
    
    def load_aggregated_data(self, filename: str = "aggregated_data.json") -> Optional[Dict[str, Dict]]:
        """
        Загрузка агрегированных данных из файла
        
        Args:
            filename: Имя файла для загрузки
            
        Returns:
            Dict или None: Загруженные данные
        """
        input_file = os.path.join(self.output_path, filename)
        
        if not os.path.exists(input_file):
            self.logger.warning(f"Файл агрегированных данных не найден: {input_file}")
            return None
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Агрегированные данные загружены из: {input_file}")
            self.logger.info(f"Загружено программ: {len(data)}")
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных из {input_file}: {str(e)}")
            return None
    
    def generate_statistics_report(self, data: Dict[str, Dict]) -> Dict[str, any]:
        """
        Генерация отчета по агрегированным данным
        
        Args:
            data: Агрегированные данные
            
        Returns:
            Dict: Статистический отчет
        """
        total_programs = len(data)
        total_launches = sum(len(prog['program_info']['launches']) for prog in data.values())
        
        # Статистика по размерностям
        dimensions_stats = {}
        programs_with_sequential_time = 0
        
        for program_name, program_data in data.items():
            # Подсчитываем программы с последовательным временем
            if 'sequential_execution_time_sec' in program_data['program_info']:
                programs_with_sequential_time += 1
        
        report = {
            'total_programs': total_programs,
            'total_parallel_launches': total_launches,
            'programs_with_sequential_time': programs_with_sequential_time,
            'average_launches_per_program': total_launches / total_programs if total_programs > 0 else 0,
            'data_completeness': {
                'coverage_data': total_programs,
                'parallel_results': sum(1 for prog in data.values() if prog['program_info']['launches']),
                'sequential_timing': programs_with_sequential_time
            }
        }
        
        self.logger.info("Сгенерирован статистический отчет:")
        self.logger.info(f"  Всего программ: {report['total_programs']}")
        self.logger.info(f"  Всего параллельных запусков: {report['total_parallel_launches']}")
        self.logger.info(f"  Программ с последовательным временем: {report['programs_with_sequential_time']}")
        
        return report