"""
Module for working with chat and dialogues on articles
"""

from typing import Dict, List, Optional
import streamlit as st
import logging
from paper_rag import rag_pipeline
from paper_rag.async_processor import async_processor
from llm_models import llm_factory, get_best_available_model
from .dialogue_manager import article_dialogue_manager
from paper_rag.embeddings import embedding_manager

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Class for managing chat and dialogues
    """
    
    def __init__(self):
        self.chat_key = 'chat_history'
        self.llm_model = None
        self.current_model_info = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """
        Initialize LLM model
        """
        
        try:
            if 'selected_llm_model' in st.session_state:
                model_info = st.session_state.selected_llm_model
                self.current_model_info = model_info
                self.llm_model = llm_factory.create_llm(
                    model_info['type'], 
                    model_info['name']
                )
                if self.llm_model and self.llm_model.is_available:
                    logger.info(f"LLM model restored: {self.llm_model.model_name}")
                    return
            
            self.llm_model = get_best_available_model()
            if self.llm_model:
                if hasattr(self.llm_model, 'model_type'):
                    self.current_model_info = {
                        'type': self.llm_model.model_type,
                        'name': self.llm_model.model_name
                    }
                else:
                    if 'OpenAI' in type(self.llm_model).__name__:
                        self.current_model_info = {'type': 'openai', 'name': self.llm_model.model_name}
                    else:
                        self.current_model_info = {'type': 'ollama', 'name': self.llm_model.model_name}
                
                logger.info(f"LLM model initialized: {self.llm_model.model_name}")
            else:
                logger.warning("Failed to find available LLM model")
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
    
    def initialize_chat(self):
        """
        Initialize chat history if it doesn't exist
        """
        if self.chat_key not in st.session_state:
            st.session_state[self.chat_key] = []
    
    def add_message(self, role: str, content: str, arxiv_id: Optional[str] = None):
        """
        Add message to chat history
        
        Args:
            role: Sender role ('user' or 'assistant')
            content: Message content
            arxiv_id: ID статьи для диалога (опционально)
        """
        self.initialize_chat()
        
        st.session_state[self.chat_key].append({
            'role': role,
            'content': content
        })
        
        if arxiv_id:
            article_dialogue_manager.add_message(arxiv_id, role, content)
            logger.info(f"Сообщение добавлено в диалог статьи {arxiv_id}")

    def get_chat_history(self) -> List[Dict]:
        """
        Получение истории чата
        
        Returns:
            Список сообщений чата
        """
        self.initialize_chat()
        return st.session_state[self.chat_key]
    
    def get_dialogue_history(self, arxiv_id: str) -> List[Dict]:
        """
        Получение истории диалога с поддержкой суммаризации
        
        Args:
            arxiv_id: ID статьи
            
        Returns:
            Список сообщений диалога
        """
        return article_dialogue_manager.get_dialogue_for_display(arxiv_id)

    def clear_chat(self, arxiv_id: Optional[str] = None):
        """
        Очистка истории чата
        
        Args:
            arxiv_id: ID статьи для очистки диалога (опционально)
        """
        # Очищаем стандартную историю
        st.session_state[self.chat_key] = []
        
        # Очищаем диалог статьи если указан
        if arxiv_id:
            article_dialogue_manager.clear_article_dialogue(arxiv_id)
            logger.info(f"Диалог статьи {arxiv_id} очищен")
    
    def display_chat_history(self, arxiv_id: Optional[str] = None):
        """
        Отображение истории чата в интерфейсе
        
        Args:
            arxiv_id: ID статьи для отображения диалога (опционально)
        """
        # Показываем текущую модель
        if self.current_model_info:
            model_type = self.current_model_info['type']
            model_name = self.current_model_info['name']
            
            if model_type == 'openai':
                st.info(f"🤖 **Текущая модель:** OpenAI {model_name}")
            else:
                st.info(f"🤖 **Текущая модель:** Ollama {model_name}")
        
        if arxiv_id:
            dialogue_history = self.get_dialogue_history(arxiv_id)
        else:
            dialogue_history = self.get_chat_history()
        
        if dialogue_history:
            st.markdown("#### 💬 История диалога")
            
            # Показываем статистику диалога если доступна
            #if arxiv_id:
            #    stats = article_dialogue_manager.get_article_stats(arxiv_id)
            #    if stats['has_summary']:
            #        st.info(f"📊 **Статистика диалога:** {stats['total_messages']} сообщений, "
            #               f"{stats['total_chars']} символов, суммаризация активна")
            
            for i, message in enumerate(dialogue_history):
                if message.get('is_summary'):
                    with st.container():
                        st.markdown("📋 **Суммаризация предыдущего диалога:**")
                        st.markdown(message['content'])
                elif message['role'] == 'user':
                    with st.container():
                        st.markdown(f"**🙋 Вы:** {message['content']}")
                else:
                    with st.container():
                        st.markdown(f"**🤖 Ассистент:** {message['content']}")
                
                if i < len(dialogue_history) - 1:
                    st.markdown("")
    
    def generate_response(self, user_input: str, article: Dict) -> str:
        """
        Генерация ответа ассистента на основе пользовательского ввода
        
        Args:
            user_input: Вопрос пользователя
            article: Информация о статье
            
        Returns:
            Ответ ассистента
        """
        arxiv_id = article.get('arxiv_id')
        rag_status = self._check_rag_status(arxiv_id)
        
        if rag_status['processed']:
            return self._generate_rag_response(user_input, article, arxiv_id)
        elif rag_status['processing']:
            return self._generate_processing_response(rag_status)
        else:
            return self._generate_simple_response()
    
    def _check_rag_status(self, arxiv_id: str) -> Dict:
        """
        Проверка статуса обработки статьи в RAG системе
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            Словарь со статусом
        """
        if not arxiv_id:
            return {'processed': False, 'processing': False}
        
        try:
            article_status = async_processor.get_article_status(arxiv_id)
            
            if article_status:
                status = article_status.get('status')
                if status == 'completed':
                    return {'processed': True, 'processing': False}
                elif status in ['queued', 'processing']:
                    return {'processed': False, 'processing': True, 'status': article_status}
            
            chunks_for_article = [
                chunk for chunk in embedding_manager.chunks_metadata 
                if chunk.get('metadata', {}).get('arxiv_id') == arxiv_id
            ]
            
            if len(chunks_for_article) > 0:
                return {'processed': True, 'processing': False}
            
            return {'processed': False, 'processing': False}
            
        except Exception as e:
            logger.error(f"Ошибка проверки RAG статуса: {e}")
            return {'processed': False, 'processing': False}
    
    def _generate_rag_response(self, user_input: str, article: Dict, arxiv_id: str) -> str:
        """
        Генерация ответа с использованием RAG системы и LLM
        
        Args:
            user_input: Вопрос пользователя
            article: Информация о статье
            arxiv_id: ID статьи
            
        Returns:
            Ответ на основе RAG + LLM
        """
        try:
            logger.info(f"Генерация RAG+LLM ответа для вопроса: {user_input}")
            
            rag_result = rag_pipeline.query_article(user_input, arxiv_id)
            
            if not rag_result['success']:
                return self._generate_simple_response()
            
            chunk = rag_result.get('chunk', {})
            section_chunks = rag_result.get('section_chunks', [chunk])
            
            if len(section_chunks) > 1:
                section_texts = []
                for sc in section_chunks:
                    text = sc.get('text', '').strip()
                    if text:
                        section_texts.append(text)
                context_text = " ".join(section_texts)
                
                max_context_length = 3000
                if len(context_text) > max_context_length:
                    top_chunk_text = chunk.get('text', '')
                    remaining_length = max_context_length - len(top_chunk_text) - 50
                    if remaining_length > 0:
                        context_text = context_text[:remaining_length] + "... [MOST RELEVANT PART]: " + top_chunk_text
                    else:
                        context_text = top_chunk_text
            else:
                context_text = chunk.get('text', '')
            
            chunk_metadata = chunk.get('metadata', {})
            
            dialogue_context = article_dialogue_manager.get_dialogue_context(arxiv_id)
            if dialogue_context:
                logger.info(f"Используется контекст диалога длиной {len(dialogue_context)} символов")
            
            if self.llm_model and self.llm_model.is_available:
                article_metadata = {
                    'title': article.get('title', ''),
                    'authors': article.get('authors', []),
                    'section': chunk_metadata.get('section', 'Unknown Section'),
                    'arxiv_id': arxiv_id
                }
                
                llm_response = self.llm_model.generate_chat_response(
                    user_input, context_text, article_metadata, dialogue_context
                )
                
                if llm_response['success']:
                    if len(section_chunks) > 1:
                        section_name = chunk_metadata.get('section', 'Unknown Section')
                        header = f"🤖 **Ответ на основе секции '{section_name}':**\n"
                    else:
                        header = "🤖 **Ответ на основе анализа статьи:**\n"
                    
                    response_parts = [
                        header,
                        llm_response['content']
                    ]
                    
                    section = chunk_metadata.get('section', 'Unknown Section')
                    source_info = f"\n📍 *Источник: {section}*"
                    response_parts.append(source_info)
                    
                    #top_chunk_info = self._format_top_chunk_debug(chunk)
                    #if top_chunk_info:
                    #    response_parts.append(top_chunk_info)
                    
                    return "\n".join(response_parts)
            
            if len(section_chunks) > 1:
                section_name = chunk_metadata.get('section', 'Unknown Section')
                header = f"📄 **Найдена релевантная информация из секции '{section_name}':**\n"
            else:
                header = "📄 **Найдена релевантная информация:**\n"
            
            response_parts = [
                header,
                context_text[:500] + "..." if len(context_text) > 500 else context_text
            ]
            
            section = chunk_metadata.get('section', 'Unknown Section')
            response_parts.append(f"\n📍 *Источник: {section}*")
            
            #top_chunk_info = self._format_top_chunk_debug(chunk)
            #if top_chunk_info:
            #    response_parts.append(top_chunk_info)
            
            return "\n".join(response_parts)
                
        except Exception as e:
            logger.error(f"Ошибка генерации RAG ответа: {e}")
            return self._generate_simple_response()
    
    def _generate_processing_response(self, rag_status: Dict) -> str:
        """
        Генерация ответа во время обработки статьи
        
        Args:
            rag_status: Статус обработки
            
        Returns:
            Сообщение о процессе обработки
        """
        status_info = rag_status.get('status', {})
        stage = status_info.get('stage', 'processing')
        progress = status_info.get('progress', 0)
        
        stage_messages = {
            'queued': 'в очереди на обработку',
            'downloading': 'скачивание PDF',
            'extracting_text': 'извлечение текста',
            'chunking': 'разбиение на фрагменты',
            'embedding': 'создание индекса',
            'processing': 'обработка'
        }
        
        stage_text = stage_messages.get(stage, 'обработка')
        
        return f"""🔄 **Статья обрабатывается для умного анализа**

📊 **Статус:** {stage_text} ({progress}%)

⏳ Пожалуйста, подождите... После завершения обработки я смогу давать более точные и детальные ответы на основе полного содержания статьи.

💡 Пока можете задавать общие вопросы о статье на основе аннотации."""
    
    def _generate_simple_response(self) -> str:
        """
        Генерация простого ответа без RAG
        
        Args:
            user_input: Вопрос пользователя
            article: Информация о статье
            
        Returns:
            Простой ответ
        """
        return "К сожалению, не получилось сделать RAG обработку статьи, попройти перезайти на страницу со статьей и подождать перед обновлением страницы"
    
    def switch_model(self, model_info: Dict) -> bool:
        """
        Переключение на другую LLM модель
        
        Args:
            model_info: Информация о модели {'type': 'openai/ollama', 'name': 'model_name'}
            
        Returns:
            True если переключение успешно
        """
        
        try:
            new_model = llm_factory.create_llm(
                model_info['type'], 
                model_info['name']
            )
            
            if new_model and new_model.is_available:
                self.llm_model = new_model
                self.current_model_info = model_info
                
                st.session_state.selected_llm_model = model_info
                
                logger.info(f"Переключение на модель: {model_info['type']}/{model_info['name']}")
                return True
            else:
                logger.error(f"Модель {model_info['type']}/{model_info['name']} недоступна")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка переключения модели: {e}")
            return False
    
    def _format_top_chunk_debug(self, chunk: Dict) -> str:
        """
        Форматирование отладочной информации о топ чанке
        
        Args:
            chunk: Самый релевантный чанк
            relevance: Анализ релевантности
            
        Returns:
            Отформатированная отладочная информация
        """
        try:
            chunk_metadata = chunk.get('metadata', {})
            chunk_text = chunk.get('text', '')
            score = chunk.get('score', 0)
            
            # Показываем отладочную информацию всегда (для отладки)
            # if relevance.get('level') not in ['низкая', 'очень низкая']:
            #     return ""
            
            debug_parts = [
                f"\n🔍 **ОТЛАДКА - Самый релевантный чанк (Score: {score:.3f}):**",
                f"📂 Секция: {chunk_metadata.get('section', 'Unknown')}",
                f"📝 Чанк #{chunk_metadata.get('chunk_index', 'N/A')}",
                f"🔍 Тип поиска: {chunk.get('search_type', 'standard')}",
            ]
            
            # Добавляем информацию о гибридном поиске если доступна
            if chunk.get('search_type') == 'hybrid':
                bm25_score = chunk.get('bm25_score', 0)
                semantic_score = chunk.get('semantic_score', 0)
                debug_parts.append(f"⚡ BM25 score: {bm25_score:.3f}, Semantic score: {semantic_score:.3f}")
            elif chunk.get('search_type') == 'bm25':
                debug_parts.append(f"📝 BM25 поиск (лексический)")
            else:
                debug_parts.append(f"🧠 Эмбеддинг поиск (семантический)")
            
            # Показываем BM25 кандидатов если есть
            bm25_candidates = chunk.get('debug_bm25_candidates', [])
            if bm25_candidates:
                debug_parts.append(f"\n📋 **BM25 кандидаты (топ-{len(bm25_candidates)}):**")
                for i, candidate in enumerate(bm25_candidates, 1):
                    cand_meta = candidate.get('metadata', {})
                    cand_section = cand_meta.get('section', 'Unknown')
                    cand_chunk = cand_meta.get('chunk_index', 'N/A')
                    cand_score = candidate.get('score', 0)
                    cand_text = candidate.get('text', '')[:100].replace('\n', ' ')
                    debug_parts.append(f"   {i}. BM25: {cand_score:.3f} | Чанк #{cand_chunk} | {cand_section}")
                    debug_parts.append(f"      Текст: {cand_text}...")
            
            # Показываем полный текст выбранного чанка
            debug_parts.append(f"\n💬 **Полный текст выбранного чанка:**")
            debug_parts.append(f"{chunk_text}")
            
            return "\n".join(debug_parts)
            
        except Exception as e:
            logger.error(f"Ошибка форматирования отладочной информации: {e}")
            return ""

chat_manager = ChatManager()
