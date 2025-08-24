"""
Конфигурация и настройки для LLM моделей
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

def get_ollama_host() -> str:
    """
    Автоматически определяет правильный хост для Ollama
    
    Логика определения:
    1. Если установлена переменная окружения OLLAMA_HOST - используем её
    2. Если запущены в Docker (есть /.dockerenv или DOCKER_ENV=true) - используем 'ollama:11434'
    3. Иначе (локальный запуск) - используем 'localhost:11434'
    
    Returns:
        Строка с URL для подключения к Ollama
    """
    # Если указан в переменной окружения, используем его
    if os.getenv('OLLAMA_HOST'):
        return os.getenv('OLLAMA_HOST')
    
    # Проверяем, запущены ли мы в Docker
    if os.path.exists('/.dockerenv') or os.getenv('DOCKER_ENV'):
        return 'http://ollama:11434'  # В Docker используем имя сервиса
    else:
        return 'http://localhost:11434'  # Локально используем localhost

# Конфигурация OpenAI
OPENAI_CONFIG = {
    'api_key': os.getenv('OPENAI_API_KEY', ''),
    'model': 'gpt-3.5-turbo',
    'temperature': 0.7,
    'max_tokens': 2000,
    'timeout': 30
}

# Конфигурация Ollama
OLLAMA_CONFIG = {
    'host': get_ollama_host(),
    'model': 'qwen3:latest',
    'temperature': 0.7,
    'timeout': 300 
}

GENERAL_CONFIG = {
    'preferred_provider': os.getenv('LLM_PROVIDER', 'ollama').lower(),
    'max_retries': 3,
    'retry_delay': 1
}

# Путь к файлу конфигурации
CONFIG_FILE = Path(__file__).parent.parent / '.env'

class LLMConfig:
    """
    Класс для управления конфигурацией LLM моделей
    """
    
    DEFAULT_CONFIG = {
        'openai': {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 2048,
            'timeout': 30
        },
        'ollama': {
            'model': 'llama2',
            'host': get_ollama_host(),
            'temperature': 0.7,
            'max_tokens': 2048,
            'timeout': 300
        },
        'general': {
            'preferred_provider': 'auto',
            'fallback_enabled': True,
            'cache_responses': False
        }
    }
    
    def __init__(self):
        """
        Инициализация конфигурации
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_from_env()
    
    def load_from_env(self):
        """
        Загрузка конфигурации из переменных окружения
        """
        if os.getenv('OPENAI_API_KEY'):
            self.config['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('OPENAI_MODEL'):
            self.config['openai']['model'] = os.getenv('OPENAI_MODEL')
        
        if os.getenv('OPENAI_BASE_URL'):
            self.config['openai']['base_url'] = os.getenv('OPENAI_BASE_URL')
        
        if os.getenv('OLLAMA_HOST'):
            self.config['ollama']['host'] = os.getenv('OLLAMA_HOST')
        
        if os.getenv('OLLAMA_MODEL'):
            self.config['ollama']['model'] = os.getenv('OLLAMA_MODEL')
        
        if os.getenv('LLM_PROVIDER'):
            self.config['general']['preferred_provider'] = os.getenv('LLM_PROVIDER')
        
        if os.getenv('LLM_TEMPERATURE'):
            try:
                temp = float(os.getenv('LLM_TEMPERATURE'))
                self.config['openai']['temperature'] = temp
                self.config['ollama']['temperature'] = temp
            except ValueError:
                pass
    
    def get_openai_config(self) -> Dict[str, Any]:
        """
        Получение конфигурации для OpenAI
        
        Returns:
            Словарь с настройками OpenAI
        """
        return self.config['openai'].copy()
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """
        Получение конфигурации для Ollama
        
        Returns:
            Словарь с настройками Ollama
        """
        return self.config['ollama'].copy()
    
    def get_general_config(self) -> Dict[str, Any]:
        """
        Получение общих настроек
        
        Returns:
            Словарь с общими настройками
        """
        return self.config['general'].copy()
    
    def set_openai_api_key(self, api_key: str):
        """
        Установка API ключа для OpenAI
        
        Args:
            api_key: API ключ
        """
        self.config['openai']['api_key'] = api_key
        os.environ['OPENAI_API_KEY'] = api_key
    
    def set_preferred_provider(self, provider: str):
        """
        Установка предпочитаемого провайдера
        
        Args:
            provider: Провайдер ('openai', 'ollama', 'auto')
        """
        if provider in ['openai', 'ollama', 'auto']:
            self.config['general']['preferred_provider'] = provider
    
    def create_env_template(self) -> str:
        """
        Создание шаблона .env файла
        
        Returns:
            Содержимое шаблона
        """
        template = """# LLM Models Configuration

# OpenAI Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
# OPENAI_BASE_URL=https://api.openai.com/v1  # Для совместимых API

# Ollama Settings
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=llama2

# General LLM Settings
LLM_PROVIDER=auto  # auto, openai, ollama
LLM_TEMPERATURE=0.7

# RAG Settings (if needed)
# RAG_CHUNK_SIZE=1000
# RAG_CHUNK_OVERLAP=200
"""
        return template
    
    def save_env_template(self, file_path: str = None):
        """
        Сохранение шаблона .env файла
        
        Args:
            file_path: Путь для сохранения (по умолчанию в корне проекта)
        """
        if file_path is None:
            file_path = Path(__file__).parent.parent / '.env.example'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.create_env_template())
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Валидация текущей конфигурации
        
        Returns:
            Результат валидации
        """
        issues = []
        warnings = []
        
        # Проверка OpenAI
        if 'api_key' not in self.config['openai']:
            issues.append("OpenAI API ключ не установлен")
        
        # Проверка Ollama
        ollama_host = self.config['ollama']['host']
        if not ollama_host.startswith('http'):
            warnings.append("Ollama host должен начинаться с http:// или https://")
        
        # Проверка температуры
        for provider in ['openai', 'ollama']:
            temp = self.config[provider]['temperature']
            if not 0 <= temp <= 2:
                warnings.append(f"Температура {provider} должна быть между 0 и 2")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'config': self.config
        }

# Глобальный экземпляр конфигурации
llm_config = LLMConfig()
