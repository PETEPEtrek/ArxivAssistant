"""
Модуль для управления диалогом с автоматической суммаризацией
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DialogueMessage:
    """
    Класс для представления сообщения в диалоге
    """
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        Инициализация сообщения
        
        Args:
            role: Роль ('user' или 'assistant')
            content: Содержание сообщения
            timestamp: Временная метка
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.char_count = len(content)
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'char_count': self.char_count
        }
    
    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}..."

class DialogueManager:
    """
    Менеджер диалога с автоматической суммаризацией
    """
    
    def __init__(self, max_chars: int = 1000):
        """
        Инициализация менеджера диалога
        
        Args:
            max_chars: Максимальное количество символов перед суммаризацией
        """
        self.max_chars = max_chars
        self.messages: List[DialogueMessage] = []
        self.summary_block: Optional[str] = None
        self.total_chars = 0
        
    def add_message(self, role: str, content: str) -> None:
        """
        Добавление нового сообщения
        
        Args:
            role: Роль отправителя
            content: Содержание сообщения
        """
        message = DialogueMessage(role, content)
        self.messages.append(message)
        self.total_chars += message.char_count
        
        # Проверяем необходимость суммаризации
        if self.total_chars > self.max_chars:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self) -> None:
        """
        Суммаризация старых сообщений
        """
        if len(self.messages) < 2:
            return
        
        # Берем первую половину сообщений для суммаризации
        half_index = len(self.messages) // 2
        messages_to_summarize = self.messages[:half_index]
        
        # Создаем текст для суммаризации
        summary_text = self._create_summary_text(messages_to_summarize)
        
        # Добавляем к существующему суммаризационному блоку
        if self.summary_block:
            self.summary_block += f"\n\n**ПРЕДЫДУЩИЙ ДИАЛОГ:**\n{summary_text}"
        else:
            self.summary_block = f"**ПРЕДЫДУЩИЙ ДИАЛОГ:**\n{summary_text}"
        
        # Удаляем суммаризированные сообщения
        self.messages = self.messages[half_index:]
        
        # Пересчитываем общее количество символов
        self.total_chars = sum(msg.char_count for msg in self.messages)
        
        logger.info(f"Суммаризировано {half_index} сообщений, осталось {len(self.messages)}")
    
    def _create_summary_text(self, messages: List[DialogueMessage]) -> str:
        """
        Создание текста для суммаризации
        
        Args:
            messages: Список сообщений для суммаризации
            
        Returns:
            Текст для суммаризации
        """
        summary_parts = []
        
        for i, message in enumerate(messages):
            role_emoji = "🙋" if message.role == 'user' else "🤖"
            summary_parts.append(f"{role_emoji} **{message.role.title()}:** {message.content}")
            
            if i < len(messages) - 1:
                summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def get_dialogue_context(self) -> str:
        """
        Получение контекста диалога для промпта
        
        Returns:
            Контекст диалога
        """
        context_parts = []
        
        # Добавляем суммаризационный блок если есть
        if self.summary_block:
            context_parts.append(self.summary_block)
            context_parts.append("")
        
        # Добавляем текущие сообщения
        if self.messages:
            context_parts.append("**ТЕКУЩИЙ ДИАЛОГ:**")
            for message in self.messages:
                role_emoji = "🙋" if message.role == 'user' else "🤖"
                context_parts.append(f"{role_emoji} **{message.role.title()}:** {message.content}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """
        Получение статистики диалога
        
        Returns:
            Статистика диалога
        """
        return {
            'total_messages': len(self.messages),
            'total_chars': self.total_chars,
            'has_summary': bool(self.summary_block),
            'summary_length': len(self.summary_block) if self.summary_block else 0,
            'max_chars': self.max_chars
        }
    
    def clear_dialogue(self) -> None:
        """
        Очистка диалога и суммаризационного блока
        """
        self.messages.clear()
        self.summary_block = None
        self.total_chars = 0
        logger.info("Диалог очищен")
    
    def get_recent_messages(self, count: int = 5) -> List[DialogueMessage]:
        """
        Получение последних сообщений
        
        Args:
            count: Количество сообщений
            
        Returns:
            Список последних сообщений
        """
        return self.messages[-count:] if self.messages else []
    
    def get_dialogue_for_display(self) -> List[Dict]:
        """
        Получение диалога для отображения в UI
        
        Returns:
            Список сообщений для отображения
        """
        display_messages = []
        
        # Добавляем суммаризационный блок если есть
        if self.summary_block:
            display_messages.append({
                'role': 'summary',
                'content': self.summary_block,
                'timestamp': None,
                'is_summary': True
            })
        
        # Добавляем текущие сообщения
        for message in self.messages:
            display_messages.append({
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp,
                'is_summary': False
            })
        
        return display_messages

class ArticleDialogueManager:
    """
    Менеджер диалогов для конкретных статей
    """
    
    def __init__(self):
        """
        Инициализация менеджера статей
        """
        self.article_dialogues: Dict[str, DialogueManager] = {}
    
    def get_dialogue_manager(self, arxiv_id: str) -> DialogueManager:
        """
        Получение менеджера диалога для статьи
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            Менеджер диалога
        """
        if arxiv_id not in self.article_dialogues:
            self.article_dialogues[arxiv_id] = DialogueManager()
        
        return self.article_dialogues[arxiv_id]
    
    def add_message(self, arxiv_id: str, role: str, content: str) -> None:
        """
        Добавление сообщения в диалог статьи
        
        Args:
            arxiv_id: ID статьи
            role: Роль отправителя
            content: Содержание сообщения
        """
        dialogue_manager = self.get_dialogue_manager(arxiv_id)
        dialogue_manager.add_message(role, content)
    
    def get_dialogue_context(self, arxiv_id: str) -> str:
        """
        Получение контекста диалога для статьи
        
        Args:
            arxiv_id: ID статьи
            
        Returns:
            Контекст диалога
        """
        dialogue_manager = self.get_dialogue_manager(arxiv_id)
        return dialogue_manager.get_dialogue_context()
    
    def clear_article_dialogue(self, arxiv_id: str) -> None:
        """
        Очистка диалога для конкретной статьи
        
        Args:
            arxiv_id: ID статьи
        """
        if arxiv_id in self.article_dialogues:
            self.article_dialogues[arxiv_id].clear_dialogue()
            logger.info(f"Диалог для статьи {arxiv_id} очищен")
    
    def get_article_stats(self, arxiv_id: str) -> Dict:
        """
        Получение статистики диалога для статьи
        
        Args:
            arxiv_id: ID статьи
            
        Returns:
            Статистика диалога
        """
        dialogue_manager = self.get_dialogue_manager(arxiv_id)
        return dialogue_manager.get_stats()
    
    def get_dialogue_for_display(self, arxiv_id: str) -> List[Dict]:
        """
        Получение диалога для отображения
        
        Args:
            arxiv_id: ID статьи
            
        Returns:
            Список сообщений для отображения
        """
        dialogue_manager = self.get_dialogue_manager(arxiv_id)
        return dialogue_manager.get_dialogue_for_display()

# Глобальный экземпляр менеджера статей
article_dialogue_manager = ArticleDialogueManager()
