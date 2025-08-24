"""
Базовый абстрактный класс для LLM моделей
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """
    Абстрактный базовый класс для всех LLM моделей
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Инициализация базового LLM
        
        Args:
            model_name: Название модели
            **kwargs: Дополнительные параметры
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_available = False
        
    @abstractmethod
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: float = 0.7) -> Dict:
        """
        Генерация ответа от модели
        
        Args:
            prompt: Входной промпт
            context: Дополнительный контекст
            max_tokens: Максимальное количество токенов
            temperature: Температура для генерации
            
        Returns:
            Словарь с ответом и метаданными
        """
        pass
    
    @abstractmethod
    def check_availability(self) -> bool:
        """
        Проверка доступности модели
        
        Returns:
            True если модель доступна
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Получение информации о модели
        
        Returns:
            Словарь с информацией о модели
        """
        pass
    
    @abstractmethod
    def generate_chat_response(self, 
                              user_question: str, 
                              article_context: str, 
                              article_metadata: Dict,
                              dialogue_context: Optional[str] = None) -> Dict:
        """
        Генерация ответа для чата по статье
        
        Args:
            user_question: Вопрос пользователя
            article_context: Релевантный контекст из статьи
            article_metadata: Метаданные статьи
            dialogue_context: Контекст предыдущего диалога (опционально)
            
        Returns:
            Ответ модели
        """
        pass
    
    def format_chat_prompt(self, 
                          user_question: str, 
                          article_context: str, 
                          article_metadata: Dict,
                          dialogue_context: Optional[str] = None) -> str:
        """
        Форматирование промпта для чата по статье
        
        Args:
            user_question: Вопрос пользователя
            article_context: Контекст из статьи
            article_metadata: Метаданные статьи
            dialogue_context: Контекст предыдущего диалога (опционально)
            
        Returns:
            Отформатированный промпт
        """
        article_title = article_metadata.get('title', 'Неизвестная статья')
        authors = article_metadata.get('authors', [])
        authors_text = ', '.join(authors) if authors else 'Неизвестные авторы'
        section = article_metadata.get('section', 'Неизвестный раздел')
        
        prompt_parts = [
            f"You are the professional arxiv paper reviewer. You are given a question and a relevant context from the article. You need to answer the question based on the context. You are also given the article title, authors, and section. You are also given the dialogue history if there is any.",
            "",
            f"ARTICLE: \"{article_title}\"",
            f"AUTHORS: {authors_text}",
            f"SECTION: {section}",
            "",
            "RELEVANT CONTEXT FROM THE ARTICLE:",
            article_context
        ]
        
        # Добавляем контекст диалога если есть
        if dialogue_context:
            prompt_parts.extend([
                "",
                "DIALOGUE HISTORY:",
                dialogue_context
            ])
        
        prompt_parts.extend([
            "",
            f"USER'S QUERY: {user_question}",
            "",
            "INSTRUCTIONS:",
            "1. Answer accurately based on the provided context",
            "2. If information is insufficient, honestly say so",
            "3. Use professional but understandable language",
            "4. Provide specific quotes or references to text parts when possible",
            "5. If the question concerns details not in the context, suggest referring to the full article text",
            "6. Consider previous dialogue when forming the answer",
            "",
            "ANSWER:"
        ])
        
        return "\n".join(prompt_parts)
    
    def format_summary_prompt(self, article_context: str, article_metadata: Dict) -> str:
        """
        Форматирование промпта для краткого изложения
        
        Args:
            article_context: Контекст из статьи
            article_metadata: Метаданные статьи
            
        Returns:
            Отформатированный промпт
        """
        article_title = article_metadata.get('title', 'Неизвестная статья')
        section = article_metadata.get('section')
        total_chunks = article_metadata.get('total_chunks', 1)
        
        # Если это суммаризация конкретного раздела
        if section and section != 'Неизвестный раздел':
            prompt = f"""Create a brief and structured summary of a scientific article section.

ARTICLE: "{article_title}"
SECTION: "{section}"
VOLUME: {total_chunks} parts

SECTION CONTENT:
{article_context}

Create a brief summary of this section in the following format:

**🎯 KEY POINTS:**
[2-3 sentences about the main ideas of this section]

**📋 CONTENT:**
[Detailed but concise description of what is covered in the section]

**💡 IMPORTANT FINDINGS:**
[Main conclusions and findings from this section]

**🔗 CONNECTION TO RESEARCH:**
[How this section relates to the overall research goal]

REQUIREMENTS:
- Answer specifically based on the content of this section
- Use simple scientific language
- Preserve important technical details
- Be brief but informative (2-4 paragraphs)"""
        else:
            # Общая суммаризация статьи
            prompt = f"""Create a brief and structured summary of a scientific article.

ARTICLE: "{article_title}"

CONTENT:
{article_context}

Create a brief summary in the following format:

**🎯 MAIN IDEA:**
[1-2 sentences about the main idea of the article]

**🔬 METHODS:**
[Brief description of the methods used]

**📊 RESULTS:**
[Key results and findings]

**💡 CONCLUSIONS:**
[Main conclusions of the authors]

**🎪 SIGNIFICANCE:**
[Why this research is important]

Answer briefly and to the point, use simple scientific language."""
        
        return prompt
    
    def validate_response(self, response: Dict) -> bool:
        """
        Валидация ответа от модели
        
        Args:
            response: Ответ от модели
            
        Returns:
            True если ответ валиден
        """
        required_fields = ['content', 'success']
        return all(field in response for field in required_fields)
    
    def handle_error(self, error: Exception, context: str = "") -> Dict:
        """
        Обработка ошибок модели
        
        Args:
            error: Исключение
            context: Контекст ошибки
            
        Returns:
            Словарь с информацией об ошибке
        """
        logger.error(f"Ошибка в {self.model_name} {context}: {error}")
        
        return {
            'success': False,
            'content': '',
            'error': str(error),
            'model': self.model_name,
            'context': context
        }