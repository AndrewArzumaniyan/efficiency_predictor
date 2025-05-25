import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DVMHEfficiencyMLP(nn.Module):
    """
    Многослойный перцептрон для предсказания эффективности DVMH программ
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(DVMHEfficiencyMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class DVMHAttentionModel(nn.Module):
    """
    Модель с механизмом внимания для предсказания эффективности DVMH программ
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], attention_dim=64, dropout_rate=0.3):
        super(DVMHAttentionModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.query = nn.Linear(hidden_dims[0], attention_dim)
        self.key = nn.Linear(hidden_dims[0], attention_dim)
        self.value = nn.Linear(hidden_dims[0], attention_dim)
        
        layers = []
        prev_dim = attention_dim
        
        for dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.output_layers = nn.Sequential(*layers)
        
    def attention(self, embedded):
        q = self.query(embedded).unsqueeze(1)
        k = self.key(embedded).unsqueeze(2)
        v = self.value(embedded)
        
        attention_weights = torch.bmm(q, k).squeeze()
        attention_weights = torch.sigmoid(attention_weights)
        
        attended = attention_weights.unsqueeze(1) * v
        
        return attended, attention_weights
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.bn1(embedded)
        embedded = nn.functional.relu(embedded)
        
        attended, attention_weights = self.attention(embedded)
        
        output = self.output_layers(attended)
        
        return output.squeeze(), attention_weights


class DVMHModelTrainer:
    """Класс для обучения моделей предсказания эффективности DVMH программ"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация тренера моделей
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)
        
        # Настройки из конфига
        model_config = self.config.get('model_training', {})
        self.models_dir = model_config.get('models_directory', './models')
        self.cv_config = model_config.get('cross_validation', {})
        
        # Создаем директорию для моделей
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Настройка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Используется устройство: {self.device}")
        
        self.logger.info("DVMHModelTrainer инициализирован")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return {}
    
    def prepare_data_by_programs(self, df: pd.DataFrame, target_column: str = 'target_speedup', 
                                test_programs_count: int = 3, val_size: float = 0.2) -> Dict:
        """
        Подготовка данных с разделением по программам
        
        Args:
            df: Исходный датафрейм
            target_column: Название целевой переменной
            test_programs_count: Количество программ для тестирования
            val_size: Доля валидационной выборки
            
        Returns:
            Dict: Подготовленные данные
        """
        self.logger.info("Подготовка данных с разделением по программам...")
        
        df_copy = df.copy()
        
        if 'program_name' not in df_copy.columns:
            raise ValueError("Колонка 'program_name' не найдена в данных")
        
        nan_count = df_copy.isna().sum().sum()
        self.logger.info(f"Количество пропущенных значений до обработки: {nan_count}")
        
        unique_programs = df_copy['program_name'].unique()
        self.logger.info(f"Всего уникальных программ: {len(unique_programs)}")
        
        # Выбираем случайные программы для тестирования
        random.seed(42)  # Для воспроизводимости
        test_programs = random.sample(list(unique_programs), test_programs_count)
        self.logger.info(f"Выбраны программы для тестирования: {test_programs}")
        
        # Разделяем данные
        test_mask = df_copy['program_name'].isin(test_programs)
        df_test = df_copy[test_mask].copy()
        df_train_val = df_copy[~test_mask].copy()
        
        self.logger.info(f"Размер тренировочной выборки: {df_train_val.shape[0]} объектов")
        self.logger.info(f"Размер тестовой выборки: {df_test.shape[0]} объектов")
        
        # Обрабатываем данные
        df_train_val, df_test = self._preprocess_data(df_train_val, df_test)
        
        # Убираем ненужные колонки
        drop_columns = ['parallel_execution_time', 'launch_config', 
                       'normalized_parallel_time', 'efficiency', 'program_name']
        for col in drop_columns:
            if col in df_train_val.columns:
                df_train_val = df_train_val.drop(columns=[col])
            if col in df_test.columns:
                df_test = df_test.drop(columns=[col])
        
        # Разделяем на признаки и целевую переменную
        if target_column not in df_train_val.columns:
            raise ValueError(f"Целевая переменная {target_column} не найдена в данных")
        
        X_train_val = df_train_val.drop(columns=[target_column])
        y_train_val = df_train_val[target_column]
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]
        
        # Обрабатываем пропущенные значения
        imputer = SimpleImputer(strategy='median')
        
        # Заполняем пропущенные значения в целевой переменной
        if y_train_val.isna().any():
            target_median = y_train_val.median()
            y_train_val = y_train_val.fillna(target_median)
            y_test = y_test.fillna(target_median)
        
        # Обрабатываем признаки
        X_train_val = pd.DataFrame(imputer.fit_transform(X_train_val), columns=X_train_val.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
        # Разделяем на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=42
        )
        
        # Нормализуем данные
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Преобразуем в нужный формат
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        y_train = y_train.values.astype(np.float32)
        y_val = y_val.values.astype(np.float32)
        y_test = y_test.values.astype(np.float32)
        
        # Финальная проверка на NaN
        for name, data in [('X_train', X_train_scaled), ('X_val', X_val_scaled), 
                          ('X_test', X_test_scaled), ('y_train', y_train), 
                          ('y_val', y_val), ('y_test', y_test)]:
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                self.logger.warning(f"NaN в {name}: {nan_count}. Заменяем на 0.")
                data = np.nan_to_num(data, nan=0.0)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train_val.columns.tolist(),
            'scaler': scaler,
            'test_programs': test_programs
        }
    
    def _preprocess_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Предобработка данных"""
        
        # Обрабатываем категориальные признаки
        categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
        if 'program_name' in categorical_columns:
            categorical_columns.remove('program_name')
        
        for col in categorical_columns:
            # Заполняем NaN
            df_train[col] = df_train[col].fillna("unknown")
            df_test[col] = df_test[col].fillna("unknown")
            
            # Кодируем категории
            if df_train[col].dtype == 'object':
                train_categories = df_train[col].astype('category').cat.categories
                df_train[col] = df_train[col].astype('category').cat.codes
                
                # Для тестовых данных
                test_categories = set(df_test[col].dropna().unique())
                new_categories = test_categories - set(train_categories)
                if new_categories:
                    self.logger.info(f"Новые категории в тестовых данных для {col}: {new_categories}")
                    df_test[col] = df_test[col].apply(lambda x: "unknown" if x in new_categories else x)
                
                df_test[col] = df_test[col].map(lambda x: 
                                              list(train_categories).index(x) if x in train_categories else -1)
                df_test[col] = df_test[col].replace(-1, df_train[col].median())
        
        # Обрабатываем смешанные типы
        for col in df_train.columns:
            if df_train[col].dtype == 'object' and col != 'program_name':
                try:
                    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
                    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
                except:
                    df_train[col] = df_train[col].fillna("unknown").astype('category').cat.codes
                    train_categories = df_train[col].astype('category').cat.categories
                    df_test[col] = df_test[col].map(lambda x: 
                                                   list(train_categories).index(x) if x in train_categories else -1)
                    df_test[col] = df_test[col].replace(-1, df_train[col].median())
        
        return df_train, df_test
    
    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, 
                   batch_size: int = 128, learning_rate: float = 0.001, 
                   patience: int = 10, use_attention: bool = False) -> Tuple[nn.Module, Dict]:
        """
        Обучение модели
        
        Args:
            model: Модель для обучения
            X_train, y_train: Тренировочные данные
            X_val, y_val: Валидационные данные
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            patience: Терпение для ранней остановки
            use_attention: Использовать ли механизм внимания
            
        Returns:
            Tuple: Обученная модель и история обучения
        """
        self.logger.info("Начало обучения модели...")
        
        model = model.to(self.device)
        
        # Создаем DataLoader'ы
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # История обучения
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_losses = []
            train_predictions = []
            train_targets = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if use_attention:
                    predictions, _ = model(batch_X)
                else:
                    predictions = model(batch_X)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_predictions.extend(predictions.detach().cpu().numpy())
                train_targets.extend(batch_y.detach().cpu().numpy())
            
            # Валидация
            model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    if use_attention:
                        predictions, _ = model(batch_X)
                    else:
                        predictions = model(batch_X)
                    
                    loss = criterion(predictions, batch_y)
                    
                    val_losses.append(loss.item())
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Вычисляем метрики
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            
            # Логирование
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                               f"train_r2={train_r2:.4f}, val_r2={val_r2:.4f}")
            
            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Ранняя остановка на эпохе {epoch}")
                    break
        
        # Загружаем лучшие веса
        model.load_state_dict(best_model_state)
        
        self.logger.info("Обучение завершено")
        return model, history
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, 
                      use_attention: bool = False) -> Dict:
        """Оценка модели"""
        self.logger.info("Оценка модели...")
        
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            if use_attention:
                predictions, attention_weights = model(X_test_tensor)
                attention_weights = attention_weights.cpu().numpy()
            else:
                predictions = model(X_test_tensor)
                attention_weights = None
            
            predictions = predictions.cpu().numpy()
        
        # Вычисляем метрики
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        if use_attention:
            return metrics, attention_weights
        else:
            return metrics
    
    def plot_learning_curves(self, history: Dict) -> plt.Figure:
        """Построение кривых обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # R2 curves
        ax2.plot(history['train_r2'], label='Train R²')
        ax2.plot(history['val_r2'], label='Validation R²')
        ax2.set_title('Model R² Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, model: nn.Module, X_test: np.ndarray, 
                                  y_test: np.ndarray, use_attention: bool = False) -> plt.Figure:
        """Построение графика предсказания vs реальные значения"""
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            if use_attention:
                predictions, _ = model(X_test_tensor)
            else:
                predictions = model(X_test_tensor)
            
            predictions = predictions.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_test, predictions, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Speedup')
        ax.set_ylabel('Predicted Speedup')
        ax.set_title('Predicted vs Actual Speedup')
        ax.grid(True)
        
        # Добавляем метрики на график
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def analyze_feature_importance(self, model: nn.Module, feature_names: List[str]) -> Tuple[pd.DataFrame, plt.Figure]:
        """Анализ важности признаков (простая версия на основе весов)"""
        
        # Получаем веса первого слоя
        first_layer_weights = None
        for name, param in model.named_parameters():
            if 'model.0.weight' in name or 'embedding.weight' in name:
                first_layer_weights = param.data.abs().mean(dim=0).cpu().numpy()
                break
        
        if first_layer_weights is None:
            self.logger.warning("Не удалось получить веса для анализа важности признаков")
            return pd.DataFrame(), plt.figure()
        
        # Создаем DataFrame с важностью признаков
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': first_layer_weights
        }).sort_values('importance', ascending=False)
        
        # Строим график
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = importance_df.head(20)
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importance')
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        return importance_df, fig
    
    def save_model(self, model: nn.Module, model_name: str, metadata: Dict = None) -> str:
        """Сохранение модели"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Сохраняем метаданные
        if metadata:
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Модель сохранена: {model_path}")
        return model_path
    
    def run_full_training(self, df: pd.DataFrame, target_column: str = 'target_speedup',
                         test_programs_count: int = 3, val_size: float = 0.2,
                         hidden_dims: List[int] = [256, 128, 64], 
                         attention_dim: int = 64,
                         epochs: int = 100, 
                         batch_size: int = 128, 
                         learning_rate: float = 0.001,
                         patience: int = 10) -> Dict:
        """
        Полный цикл обучения моделей
        
        Returns:
            Dict: Результаты экспериментов
        """
        self.logger.info("Запуск полного цикла обучения моделей")
        
        # Анализ данных
        self.logger.info(f"Размерность данных: {df.shape}")
        self.logger.info(f"Числовые признаки: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        self.logger.info(f"Категориальные признаки: {len(df.select_dtypes(include=['object']).columns)}")
        
        # Подготовка данных
        data = self.prepare_data_by_programs(
            df, target_column=target_column, 
            test_programs_count=test_programs_count, 
            val_size=val_size
        )
        
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        feature_names = data['feature_names']
        test_programs = data['test_programs']
        
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Validation set: {X_val.shape}")
        self.logger.info(f"Test set: {X_test.shape}")
        self.logger.info(f"Test programs: {test_programs}")
        
        input_dim = X_train.shape[1]
        
        # 1. Обучение стандартной MLP модели
        self.logger.info("Обучение стандартной MLP модели...")
        mlp_model = DVMHEfficiencyMLP(input_dim, hidden_dims)
        trained_mlp, mlp_history = self.train_model(
            mlp_model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, 
            learning_rate=learning_rate, patience=patience
        )
        
        # 2. Обучение модели с механизмом внимания
        self.logger.info("Обучение модели с механизмом внимания...")
        attention_model = DVMHAttentionModel(input_dim, hidden_dims, attention_dim)
        trained_attention_model, attention_history = self.train_model(
            attention_model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, 
            learning_rate=learning_rate, patience=patience,
            use_attention=True
        )
        
        # Оценка моделей
        self.logger.info("Оценка моделей на тестовой выборке...")
        mlp_metrics = self.evaluate_model(trained_mlp, X_test, y_test)
        attention_metrics, attention_weights = self.evaluate_model(
            trained_attention_model, X_test, y_test, use_attention=True
        )
        
        # Вывод метрик
        self.logger.info("Метрики стандартной MLP модели:")
        for metric, value in mlp_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        self.logger.info("Метрики модели с механизмом внимания:")
        for metric, value in attention_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Создание визуализаций
        results_dir = os.path.join(self.models_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Кривые обучения
        fig1 = self.plot_learning_curves(mlp_history)
        fig1.savefig(os.path.join(results_dir, 'mlp_learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = self.plot_learning_curves(attention_history)
        fig2.savefig(os.path.join(results_dir, 'attention_learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Графики предсказаний
        fig3 = self.plot_predictions_vs_actual(trained_mlp, X_test, y_test)
        fig3.savefig(os.path.join(results_dir, 'mlp_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        fig4 = self.plot_predictions_vs_actual(trained_attention_model, X_test, y_test, use_attention=True)
        fig4.savefig(os.path.join(results_dir, 'attention_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # Анализ важности признаков
        importance, fig5 = self.analyze_feature_importance(trained_mlp, feature_names)
        fig5.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        self.logger.info("Важность признаков (топ-10):")
        for idx, row in importance.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Сохранение моделей
        mlp_metadata = {
            'model_type': 'MLP',
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'metrics': mlp_metrics,
            'test_programs': test_programs,
            'feature_names': feature_names
        }
        
        attention_metadata = {
            'model_type': 'Attention',
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'attention_dim': attention_dim,
            'metrics': attention_metrics,
            'test_programs': test_programs,
            'feature_names': feature_names
        }
        
        mlp_path = self.save_model(trained_mlp, 'dvmh_mlp_model', mlp_metadata)
        attention_path = self.save_model(trained_attention_model, 'dvmh_attention_model', attention_metadata)
        
        # Сохранение результатов
        results = {
            'mlp_model': trained_mlp,
            'attention_model': trained_attention_model,
            'mlp_metrics': mlp_metrics,
            'attention_metrics': attention_metrics,
            'mlp_history': mlp_history,
            'attention_history': attention_history,
            'feature_importance': importance,
            'feature_names': feature_names,
            'test_programs': test_programs,
            'data_info': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'feature_count': input_dim
            },
            'model_paths': {
                'mlp': mlp_path,
                'attention': attention_path
            }
        }
        
        # Сохраняем общий отчет
        report_path = os.path.join(results_dir, 'training_report.json')
        report_data = {
            'mlp_metrics': mlp_metrics,
            'attention_metrics': attention_metrics,
            'data_info': results['data_info'],
            'test_programs': test_programs,
            'feature_names': feature_names
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результаты сохранены в: {results_dir}")
        self.logger.info("Обучение моделей завершено успешно!")
        
        return results