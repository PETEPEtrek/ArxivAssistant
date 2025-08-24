"""
LLM Models пакет для работы с различными языковыми моделями

Поддерживает:
- OpenAI API модели (GPT-3.5, GPT-4)
- Локальные модели через Ollama (LLaMA 2, Mistral, etc.)
"""

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM, OLLAMA_AVAILABLE
from .llm_factory import LLMFactory, llm_factory

__version__ = "1.0.0"

__all__ = [
    'BaseLLM',
    'OpenAILLM',
    'OllamaLLM',
    'LLMFactory',
    'llm_factory',
    'OLLAMA_AVAILABLE'
]

# Удобные функции для быстрого создания моделей
def create_openai_model(model_name: str = "gpt-3.5-turbo", **kwargs) -> OpenAILLM:
    """
    Быстрое создание OpenAI модели
    
    Args:
        model_name: Название модели
        **kwargs: Дополнительные параметры
        
    Returns:
        Экземпляр OpenAILLM
    """
    return llm_factory.create_llm('openai', model_name, **kwargs)

def create_ollama_model(model_name: str = "llama2", **kwargs) -> OllamaLLM:
    """
    Быстрое создание Ollama модели
    
    Args:
        model_name: Название модели
        **kwargs: Дополнительные параметры
        
    Returns:
        Экземпляр OllamaLLM
    """
    return llm_factory.create_llm('ollama', model_name, **kwargs)

def get_best_available_model(**kwargs) -> BaseLLM:
    """
    Получение лучшей доступной модели
    
    Args:
        **kwargs: Дополнительные параметры
        
    Returns:
        Экземпляр лучшей доступной LLM
    """
    return llm_factory.create_best_available(**kwargs)

def list_available_models() -> dict:
    """
    Получение списка доступных моделей
    
    Returns:
        Словарь с доступными моделями
    """
    return llm_factory.get_available_models()
