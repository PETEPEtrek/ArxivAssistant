"""
Модуль UI компонентов для ArXiv Assistant
"""

import streamlit as st
from typing import Dict
import logging
import os
from datetime import datetime

from ui.styles import get_article_card_style
from paper_rag.async_processor import async_processor
from paper_rag.embeddings import embedding_manager
from ui.summary import summarize_paper_by_sections
from ui.dialogue_manager import article_dialogue_manager
from paper_rag.pdf_processor import PDFProcessor
from paper_rag.chunking import TextChunker
from paper_rag.embeddings import embedding_manager
from llm_models import llm_factory
from llm_models.config import llm_config

logger = logging.getLogger(__name__)

class UIComponents:
    """
    Класс для UI компонентов приложения
    """
    
    @staticmethod
    def display_article_card(article: Dict, article_index: int):
        """
        Отображение карточки статьи
        
        Args:
            article: Словарь с информацией о статье
            article_index: Индекс статьи для уникальных ключей
        """
        with st.container():
            st.markdown(
                f"""
                <div style="{get_article_card_style()}">
                    <h4 style="color: #333; margin-bottom: 8px;">{article['title']}</h4>
                    <p style="color: #666; margin-bottom: 5px; font-style: italic;">
                        <strong>Авторы:</strong> {', '.join(article['authors']) if article['authors'] else 'Не указаны'}
                    </p>
                    <p style="color: #444; line-height: 1.4; margin-bottom: 10px;">
                        {article['abstract']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            
            with col1:
                if st.button("💬 Обсудить", key=f"discuss_{article_index}"):
                    UIComponents._clear_article_state()
                    
                    st.session_state.current_page = "article_view"
                    st.session_state.selected_article = article
                    
                    UIComponents._queue_article_processing(article)
                    
                    st.rerun()
            
            with col2:
                if article['link']:
                    st.link_button("📄 Статья", article['link'])
            
            with col3:
                if article['pdf_link']:
                    st.link_button("📁 PDF", article['pdf_link'])
    
    @staticmethod
    def display_search_form():
        """
        Отображение формы поиска
        
        Returns:
            Кортеж (search_query, max_results)
        """
        search_query = st.text_input(
            "🔍 Введите поисковый запрос:",
            placeholder="Например: machine learning, quantum computing, neural networks...",
            help="Введите ключевые слова для поиска статей в arXiv"
        )
        
        _, col2 = st.columns([3, 1])
        with col2:
            max_results = st.selectbox(
                "Количество результатов:",
                options=[5, 10, 15, 20],
                index=1
            )
        
        return search_query, max_results
    
    @staticmethod
    def display_article_header(article: Dict):
        """
        Отображение заголовка статьи на странице просмотра
        
        Args:
            article: Словарь с информацией о статье
        """
        if st.button("← Назад к поиску"):
            st.session_state.current_page = "search"
            st.rerun()
        
        st.markdown("---")
        
        st.title(article['title'])
        
        if article.get('uploaded_file'):
            st.success("📁 Загруженный PDF файл")
        else:
            st.info("🔍 Статья из arXiv")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Авторы:** {', '.join(article['authors']) if article['authors'] else 'Не указаны'}")
            
            if article['published']:
                published_date = article['published'].split('T')[0]
                st.markdown(f"**Дата публикации:** {published_date}")
            
            if article['arxiv_id']:
                st.markdown(f"**ID:** {article['arxiv_id']}")
            
            if article.get('uploaded_file'):
                if article.get('original_filename'):
                    st.markdown(f"**Оригинальный файл:** {article['original_filename']}")
                if article.get('upload_timestamp'):
                    upload_time = article['upload_timestamp'].split('T')[1][:8]
                    st.markdown(f"**Время загрузки:** {upload_time}")
        
        with col2:
            if article.get('link'):
                st.link_button("📄 Открыть статью", article['link'], use_container_width=True)
            
            if article['pdf_link']:
                if article.get('uploaded_file'):
                    st.link_button("📁 Открыть PDF", article['pdf_link'], use_container_width=True)
                else:
                    st.link_button("📁 Скачать PDF", article['pdf_link'], use_container_width=True)
    
    @staticmethod
    def display_article_abstract(article: Dict):
        """
        Отображение аннотации статьи
        
        Args:
            article: Словарь с информацией о статье
        """
        st.markdown("---")
        
        st.markdown("### 📝 Аннотация")
        if article['abstract']:
            full_abstract = article.get('full_abstract', article['abstract'])
            st.markdown(f"*{full_abstract}*")
        else:
            st.markdown("*Аннотация не доступна*")
    
    @staticmethod
    def display_summarize_section(article: Dict):
        """
        Отображение секции суммаризации статьи
        
        Args:
            article: Словарь с информацией о статье
        """
        st.markdown("---")
        
        arxiv_id = article.get('arxiv_id')
        rag_ready = False
        
        if arxiv_id:
            try:
                chunks_for_article = [
                    chunk for chunk in embedding_manager.chunks_metadata 
                    if chunk.get('metadata', {}).get('arxiv_id') == arxiv_id
                ]
                rag_ready = len(chunks_for_article) > 0
            except:
                rag_ready = False
        
        currently_summarizing = st.session_state.get('summarizing', False)
        
        if rag_ready:
            st.success(f"✅ RAG готов для суммаризации {arxiv_id} ({len(chunks_for_article) if 'chunks_for_article' in locals() else 0} чанков)")
        else:
            st.info(f"📄 RAG не готов для {arxiv_id}")
        
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if currently_summarizing:
                button_text = "⏳ Суммаризация..."
                button_disabled = True
                button_type = "secondary"
            elif not rag_ready:
                button_text = "📋 Summarize Paper (ожидание RAG)"
                button_disabled = True
                button_type = "secondary" 
            else:
                button_text = "📋 Summarize Paper"
                button_disabled = False
                button_type = "primary"
            
            if st.button(button_text, use_container_width=True, type=button_type, disabled=button_disabled):
                if article.get('arxiv_id') and rag_ready:
                    st.session_state.summarizing = True
                    st.session_state.summary_sections = None
                else:
                    st.error("❌ Невозможно суммаризировать: отсутствует arXiv ID")
        
        if st.session_state.get('summarizing', False):
            st.markdown("### 🔄 Суммаризация статьи...")
            
            progress_bar = st.progress(0)
            status_text = st.info("🚀 Запуск суммаризации...")
            
            if 'summary_sections' in st.session_state and st.session_state.summary_sections:
                st.session_state.summarizing = False
                st.success("✅ Суммаризация уже завершена!")
                st.rerun()
            else:
                try:
                    with st.spinner("🤖 Обрабатываю разделы статьи..."):
                        result = summarize_paper_by_sections(article['arxiv_id'], progress_bar, status_text)
                    
                    if result['success']:
                        st.session_state.summary_sections = result['sections']
                        st.session_state.summarizing = False
                        st.success("✅ Суммаризация завершена!")
                        st.rerun()
                    else:
                        st.session_state.summarizing = False
                        st.error(f"❌ Ошибка суммаризации: {result.get('error', 'Неизвестная ошибка')}")
                        
                except Exception as e:
                    st.session_state.summarizing = False
                    st.error(f"💥 Критическая ошибка: {str(e)}")
        
        if st.session_state.get('summary_sections'):
            st.markdown("### 📄 Суммаризация по разделам")
            
            sections = st.session_state.summary_sections
            for section in sections:
                with st.expander(f"📑 {section['title']}", expanded=False):
                    #st.markdown(f"**📊 Статистика:** {len(section['chunks'])} частей, ~{section['total_length']} символов")
                    st.markdown("**🤖 Краткое содержание:**")
                    st.markdown(section['summary'])
                    
                    if st.checkbox(f"Показать оригинальный текст", key=f"show_original_{section['title']}"):
                        st.markdown("**📝 Оригинальный текст:**")
                        st.text_area("", section['original_text'], height=200, key=f"original_{section['title']}")
    
    @staticmethod
    def display_chat_input():
        """
        Отображение поля ввода для чата
        
        Returns:
            Введенный пользователем текст или None
        """
        user_input = st.text_input(
            "Начните диалог по статье:",
            placeholder="Задайте вопрос о статье, попросите объяснить сложные моменты...",
            key="chat_input"
        )
        
        col1, _ = st.columns([1, 5])
        with col1:
            send_clicked = st.button("Отправить")
        
        return user_input if (send_clicked and user_input) else None
    
    @staticmethod
    def display_search_results(articles: list):
        """
        Отображение результатов поиска
        
        Args:
            articles: Список статей для отображения
        """
        if articles:
            st.success(f"✅ Найдено {len(articles)} статей")
            
            st.session_state.search_results = articles
            
            for i, article in enumerate(articles, 1):
                st.markdown(f"### 📖 Статья {i}")
                UIComponents.display_article_card(article, i)
                
                if i < len(articles):
                    st.markdown("---")
        else:
            st.warning("😔 По вашему запросу ничего не найдено. Попробуйте изменить поисковые термины.")
    
    @staticmethod
    def display_page_header():
        """
        Отображение заголовка приложения
        """
        st.title("📚 ArXiv Assistant")
        st.markdown("### Поиск научных статей в базе данных arXiv")
    
    @staticmethod
    def _clear_article_state():
        """
        Очистка состояния при переходе на новую статью
        """
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
        
        if 'summarizing' in st.session_state:
            st.session_state.summarizing = False
        
        if 'summary_sections' in st.session_state:
            del st.session_state.summary_sections
        
        if 'input_counter' in st.session_state:
            st.session_state.input_counter = 0
        
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith('chat_input_')]
        for key in keys_to_delete:
            del st.session_state[key]
        
        if 'rag_ready' in st.session_state:
            st.session_state.rag_ready = False
        
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith('rag_ready_')]
        for key in keys_to_delete:
            del st.session_state[key]
        
        try:
            if 'previous_arxiv_id' in st.session_state:
                article_dialogue_manager.clear_article_dialogue(st.session_state.previous_arxiv_id)
                del st.session_state.previous_arxiv_id
        except ImportError:
            pass
        
        logger.info("Состояние очищено при переходе на новую статью")
    
    @staticmethod
    def _queue_article_processing(article: Dict):
        """
        Постановка статьи в очередь на асинхронную обработку для RAG
        
        Args:
            article: Информация о статье
        """
        if not async_processor:
            logger.warning("RAG система недоступна")
            return
        
        arxiv_id = article.get('arxiv_id')
        pdf_link = article.get('pdf_link')
        
        if not arxiv_id or not pdf_link:
            logger.warning("Недостаточно данных для обработки статьи")
            return
        
        try:
            if 'selected_article' in st.session_state:
                previous_arxiv_id = st.session_state.selected_article.get('arxiv_id')
                if previous_arxiv_id and previous_arxiv_id != arxiv_id:
                    st.session_state.previous_arxiv_id = previous_arxiv_id
            
            if article.get('uploaded_file'):
                if os.path.exists(pdf_link):
                    logger.info(f"Загруженный файл {arxiv_id} найден: {pdf_link}")
                    
                    if not UIComponents._check_rag_ready(article, use_cache=False):
                        logger.info(f"Запускаем RAG обработку для загруженного файла: {arxiv_id}")
                        
                        try:
                            
                            pdf_processor = PDFProcessor()
                            extracted_data = pdf_processor.extract_text_pypdf2(pdf_link)
                            
                            if extracted_data and extracted_data.get('text'):
                                text_content = extracted_data['text']
                                logger.info(f"Текст извлечен: {len(text_content)} символов")
                                
                                chunker = TextChunker()
                                
                                chunk_data = {
                                    'text': text_content,
                                    'metadata': {
                                        'arxiv_id': arxiv_id,
                                        'pdf_path': pdf_link,
                                        'source': 'uploaded_pdf'
                                    }
                                }
                                
                                chunks = chunker.chunk_text(chunk_data)
                                
                                if chunks:
                                    logger.info(f"Создано {len(chunks)} чанков для {arxiv_id}")
                                    
                                    for chunk in chunks:
                                        chunk['metadata']['arxiv_id'] = arxiv_id
                                        chunk['metadata']['source'] = 'uploaded_pdf'
                                        chunk['metadata']['file_path'] = pdf_link
                                    
                                    embedding_manager.add_to_index(chunks)
                                    
                                    logger.info(f"RAG обработка для {arxiv_id} завершена успешно")
                                else:
                                    logger.warning(f"Не удалось создать чанки для {arxiv_id}")
                            else:
                                logger.warning(f"Не удалось извлечь текст для RAG обработки: {arxiv_id}")
                                
                        except Exception as e:
                            logger.error(f"Ошибка RAG обработки загруженного файла {arxiv_id}: {e}")
                    else:
                        logger.info(f"RAG уже готов для загруженного файла: {arxiv_id}")
                    
                    return
                else:
                    logger.error(f"Загруженный файл не найден: {pdf_link}")
                    return
            else:
                task_id = async_processor.queue_article(arxiv_id, pdf_link)
                logger.info(f"Статья {arxiv_id} поставлена в очередь обработки: {task_id}")
                
                if 'processing_tasks' not in st.session_state:
                    st.session_state.processing_tasks = {}
                st.session_state.processing_tasks[arxiv_id] = task_id
            
        except Exception as e:
            logger.error(f"Ошибка постановки статьи в очередь: {e}")
    
    @staticmethod
    def _check_rag_ready(article: Dict, use_cache: bool = True) -> bool:
        """
        Эффективная проверка готовности RAG индекса для статьи
        
        Args:
            article: Информация о статье
            use_cache: Использовать кэш для предотвращения повторных проверок
            
        Returns:
            True если RAG индекс готов
        """
        
        arxiv_id = article.get('arxiv_id')
        if not arxiv_id:
            return False
        
        if article.get('uploaded_file'):
            use_cache = False
        
        cache_key = f"rag_ready_{arxiv_id}"
        if use_cache and cache_key in st.session_state:
            return st.session_state[cache_key]
        
        try:
            chunks_for_article = [
                chunk for chunk in embedding_manager.chunks_metadata 
                if chunk.get('metadata', {}).get('arxiv_id') == arxiv_id
            ]
            
            if len(chunks_for_article) == 0:
                if use_cache:
                    st.session_state[cache_key] = False
                return False

            index_exists = hasattr(embedding_manager, 'index') and embedding_manager.index is not None
            
            rag_ready = len(chunks_for_article) > 0 and index_exists
            
            if use_cache:
                st.session_state[cache_key] = rag_ready
            
            return rag_ready
            
        except Exception as e:
            logger.debug(f"Ошибка проверки RAG для {arxiv_id}: {e}")
            if use_cache:
                st.session_state[cache_key] = False
            return False
    
    @staticmethod
    def display_rag_status_indicator(article: Dict):
        """
        Отображение индикатора статуса обработки RAG
        
        Args:
            article: Информация о статье
        """
        
        arxiv_id = article.get('arxiv_id')
        if not arxiv_id:
            return
        
        # Проверяем готовность RAG
        rag_ready = UIComponents._check_rag_ready(article)
        
        # Для загруженных файлов показываем специальный статус
        if article.get('uploaded_file'):
            if rag_ready:
                st.success(f"✅ RAG готов для {arxiv_id}")
            else:
                # Показываем прогресс обработки для загруженных файлов
                with st.spinner("🔍 Обработка PDF для умного анализа..."):
                    # Проверяем статус каждые 2 секунды
                    import time
                    time.sleep(0.1)  # Небольшая задержка для UI
                    
                    # Повторно проверяем готовность
                    rag_ready = UIComponents._check_rag_ready(article, use_cache=False)
                    
                    if rag_ready:
                        st.success(f"✅ RAG готов для {arxiv_id}")
                        st.rerun()
                    else:
                        st.info("📄 Статья будет обработана для умного анализа")
        else:
            # Для arXiv статей показываем стандартный статус
            if rag_ready:
                st.success(f"✅ RAG готов для {arxiv_id}")
            else:
                st.info(f"📄 RAG не готов для {arxiv_id}")
    
    @staticmethod
    def display_enhanced_chat_input(article: Dict = None):
        """
        Расширенное поле ввода для чата с подсказками
        
        Args:
            article: Информация о статье (опционально)
        
        Returns:
            Введенный пользователем текст или None
        """
        # Получаем статью из параметра или session_state
        if article is None:
            article = st.session_state.get('selected_article', {})
        
        # УПРОЩЕННАЯ ЛОГИКА: Проверяем RAG без кэша каждый раз
        arxiv_id = article.get('arxiv_id')
        rag_ready = False
        
        if arxiv_id:
            # Простая проверка: есть ли чанки для этой статьи
            try:
                from paper_rag.embeddings import embedding_manager
                chunks_for_article = [
                    chunk for chunk in embedding_manager.chunks_metadata 
                    if chunk.get('metadata', {}).get('arxiv_id') == arxiv_id
                ]
                rag_ready = len(chunks_for_article) > 0
            except:
                rag_ready = False
        
        
        # Простая логика активации
        if rag_ready:
            placeholder_text = "Who are the authors of this paper?"
            input_disabled = False
            # Отладочная информация
            st.success(f"✅ RAG готов для {arxiv_id}")
        else:
            st.info("📄 Статья обрабатывается для умного анализа... Попробуйте обновить страницу.")
            placeholder_text = "Ожидание обработки статьи..."
            input_disabled = True
        
        # ПРОСТАЯ ЛОГИКА ОЧИСТКИ: используем уникальный ключ каждый раз
        input_key = f"chat_input_{arxiv_id}_{st.session_state.get('input_counter', 0)}"
        
        # Поле ввода
        user_input = st.text_input(
            "Задайте вопрос о статье:",
            placeholder=placeholder_text,
            key=input_key,
            disabled=input_disabled
        )
        
        # Кнопка отправки
        col1, _ = st.columns([1, 5])
        with col1:
            # Кнопка активна только если RAG готов И есть непустой текст
            has_text = user_input and user_input.strip()
            button_disabled = input_disabled or not has_text
            
            # Отладочная информация о состоянии кнопки
            if rag_ready:
                button_text = "📤 Отправить"
                if not has_text:
                    st.caption("💡 Введите текст для активации кнопки")
            else:
                button_text = "📤 Отправить (RAG не готов)"
            
            send_clicked = st.button(
                button_text, 
                use_container_width=True,
                disabled=button_disabled
            )
        
        # Обрабатываем отправку
        if send_clicked and user_input and user_input.strip() and rag_ready:
            # Увеличиваем счетчик для создания нового поля ввода
            st.session_state.input_counter = st.session_state.get('input_counter', 0) + 1
            return user_input.strip()
        
        return None

    @staticmethod
    def display_llm_model_selector():
        """
        Отображение селектора LLM модели
        
        Returns:
            Выбранная модель или None
        """
        st.markdown("---")
        st.markdown("### 🤖 Выбор модели ИИ")
        
        # Проверяем доступность моделей
        try:
            
            # Получаем доступные модели
            available_models = llm_factory.get_available_models()
            
            # Проверяем OpenAI API ключ
            openai_api_key = llm_config.get_openai_config().get('api_key') or os.getenv('OPENAI_API_KEY')
            openai_available = openai_api_key
            
            # Проверяем Ollama - есть ли установленные модели
            ollama_models = available_models.get('ollama', [])
            ollama_available = any(model.get('installed', False) for model in ollama_models)
            
            # Создаем опции для выбора
            model_options = []
            model_descriptions = {}
            
            # OpenAI модели
            if openai_available:
                for model in available_models.get('openai', []):
                    option_key = f"openai_{model['name']}"
                    model_options.append(option_key)
                    model_descriptions[option_key] = f"OpenAI {model['name']} - {model['description']}"
            else:
                # Показываем OpenAI как недоступную
                model_options.append("openai_disabled")
                model_descriptions["openai_disabled"] = "OpenAI GPT-3.5 - Требуется API ключ"
            
            # Ollama модели
            if ollama_available:
                for model in available_models.get('ollama', []):
                    if model.get('installed', False):
                        option_key = f"ollama_{model['name']}"
                        model_options.append(option_key)
                        model_descriptions[option_key] = f"Ollama {model['name']} - {model['description']} (локальная)"
            
            # Если нет доступных моделей
            if not model_options:
                st.error("❌ Нет доступных LLM моделей")
                return None
            
            # Селектор модели
            selected_option = st.selectbox(
                "Выберите модель для диалога:",
                options=model_options,
                format_func=lambda x: model_descriptions.get(x, x),
                help="Выберите модель для генерации ответов на ваши вопросы"
            )
            
            # Обработка выбора
            if selected_option == "openai_disabled":
                st.warning("⚠️ Для использования OpenAI требуется API ключ")
                
                # Форма для ввода API ключа
                with st.expander("🔑 Настройка OpenAI API"):
                    st.info("""
                    Для использования OpenAI моделей необходимо:
                    1. Получить API ключ на [platform.openai.com](https://platform.openai.com)
                    2. Установить переменную окружения OPENAI_API_KEY
                    3. Или ввести ключ ниже (не рекомендуется для продакшена)
                    """)
                    
                    api_key = st.text_input(
                        "OpenAI API Key:",
                        type="password",
                        placeholder="sk-...",
                        help="Введите ваш OpenAI API ключ"
                    )
                    
                    if st.button("💾 Сохранить ключ"):
                        if api_key and api_key.startswith("sk-"):
                            # Сохраняем в конфигурации
                            llm_config.set_openai_api_key(api_key)
                            st.success("✅ API ключ сохранен! Обновите страницу для применения.")
                            st.rerun()
                        else:
                            st.error("❌ Неверный формат API ключа")
                
                return None
            
            elif selected_option.startswith("openai_"):
                model_name = selected_option.replace("openai_", "")
                st.success(f"✅ Выбрана модель: OpenAI {model_name}")
                
                # Показываем информацию о модели
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Модель:** {model_name}")
                    st.info(f"**Провайдер:** OpenAI")
                with col2:
                    st.info(f"**Статус:** Доступна")
                    st.info(f"**Тип:** Облачная")
                
                return {"type": "openai", "name": model_name}
            
            elif selected_option.startswith("ollama_"):
                model_name = selected_option.replace("ollama_", "")
                st.success(f"✅ Выбрана модель: Ollama {model_name}")
                
                # Показываем информацию о модели
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Модель:** {model_name}")
                    st.info(f"**Провайдер:** Ollama")
                with col2:
                    st.info(f"**Статус:** Установлена")
                    st.info(f"**Тип:** Локальная")
                
                return {"type": "ollama", "name": model_name}
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки моделей: {e}")
            return None
        
        return None

    @staticmethod
    def display_chat_history():
        """
        Отображение истории чата в интерфейсе
        """
        # Этот метод будет реализован в ChatManager
        # Здесь оставляем заглушку для совместимости
        pass

    @staticmethod
    def display_pdf_upload_section():
        """
        Отображение секции загрузки PDF файлов
        
        Returns:
            Загруженная статья или None
        """
        st.markdown("---")
        st.markdown("### 📁 Загрузить свою статью")
        
        # Информация о поддерживаемых форматах
        with st.expander("ℹ️ Информация о загрузке"):
            st.markdown("""
            **Поддерживаемые форматы:**
            - PDF файлы (.pdf)
            
            **Что происходит при загрузке:**
            1. Файл сохраняется локально
            2. Извлекается текст и метаданные
            3. Генерируется уникальный ID
            4. Статья становится доступной для анализа
            
            **Ограничения:**
            - Максимальный размер: 50 MB
            - Только PDF файлы
            - Файлы сохраняются в папке `uploaded_pdfs`
            """)
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите PDF файл:",
            type=['pdf'],
            help="Загрузите PDF файл для анализа"
        )
        
        if uploaded_file is not None:
            # Показываем информацию о загруженном файле
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📄 **Файл:** {uploaded_file.name}")
            with col2:
                st.info(f"📏 **Размер:** {file_size:.2f} MB")
            with col3:
                st.info(f"📅 **Загружен:** {datetime.now().strftime('%H:%M:%S')}")
            
            # Кнопка обработки
            if st.button("🚀 Обработать PDF", type="primary", use_container_width=True):
                with st.spinner("🔍 Обработка PDF файла..."):
                    try:
                        from ui.pdf_uploader import pdf_uploader
                        
                        # Обрабатываем загруженный файл
                        article_info = pdf_uploader.process_uploaded_pdf(uploaded_file)
                        
                        if article_info:
                            st.success(f"✅ PDF файл успешно обработан!")
                            
                            # Показываем извлеченную информацию
                            with st.expander("📋 Извлеченная информация", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**Название:** {article_info.get('title', 'Неизвестно')}")
                                    st.markdown(f"**Авторы:** {', '.join(article_info.get('authors', ['Неизвестны']))}")
                                    st.markdown(f"**ID:** {article_info.get('arxiv_id', 'Неизвестно')}")
                                
                                with col2:
                                    st.markdown(f"**Дата публикации:** {article_info.get('published', 'Неизвестно')}")
                                    st.markdown(f"**Файл:** {article_info.get('original_filename', 'Неизвестно')}")
                                    st.markdown(f"**Тип:** Загруженный PDF")
                                
                                # Аннотация
                                st.markdown("**Аннотация:**")
                                abstract = article_info.get('abstract', 'Аннотация недоступна')
                                st.markdown(f"*{abstract}*")
                            
                            # Кнопка для перехода к статье
                            if st.button("💬 Обсудить статью", type="secondary", use_container_width=True):
                                return article_info
                        else:
                            st.error("❌ Не удалось обработать PDF файл")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка обработки: {str(e)}")
                        logger.error(f"Ошибка обработки PDF: {e}")
        
        return None

    @staticmethod
    def display_uploaded_articles_section():
        """
        Отображение секции с загруженными статьями
        
        Returns:
            Выбранная статья или None
        """
        try:
            from ui.pdf_uploader import pdf_uploader
            
            # Получаем список загруженных статей
            uploaded_articles = pdf_uploader.get_uploaded_articles()
            
            if not uploaded_articles:
                return None
            
            st.markdown("---")
            st.markdown("### 📚 Загруженные статьи")
            
            # Показываем загруженные статьи
            for i, article in enumerate(uploaded_articles):
                with st.container():
                    # Создаем стилизованную карточку
                    st.markdown(
                        f"""
                        <div style="{get_article_card_style()}">
                            <h4 style="color: #333; margin-bottom: 8px;">{article['title']}</h4>
                            <p style="color: #666; margin-bottom: 5px; font-style: italic;">
                                <strong>Авторы:</strong> {', '.join(article['authors']) if article['authors'] else 'Не указаны'}
                            </p>
                            <p style="color: #444; line-height: 1.4; margin-bottom: 10px;">
                                {article['abstract']}
                            </p>
                            <p style="color: #888; font-size: 0.9em;">
                                📅 Загружено: {article.get('upload_timestamp', 'Неизвестно')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Кнопки для действий
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                    
                    with col1:
                        if st.button("💬 Обсудить", key=f"discuss_uploaded_{i}"):
                            # Очищаем состояние при переходе на новую статью
                            UIComponents._clear_article_state()
                            
                            st.session_state.current_page = "article_view"
                            st.session_state.selected_article = article
                            
                            # Запускаем асинхронную обработку статьи для RAG
                            UIComponents._queue_article_processing(article)
                            
                            st.rerun()
                    
                    with col2:
                        if article['file_path']:
                            st.link_button("📄 Открыть", article['file_path'])
                    
                    with col3:
                        if st.button("🗑️ Удалить", key=f"delete_uploaded_{i}"):
                            if pdf_uploader.delete_uploaded_article(article['arxiv_id']):
                                st.success("✅ Статья удалена")
                                st.rerun()
                            else:
                                st.error("❌ Ошибка удаления")
                    
                    # Разделитель между статьями
                    if i < len(uploaded_articles) - 1:
                        st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки списка статей: {str(e)}")
            logger.error(f"Ошибка отображения загруженных статей: {e}")
        
        return None

# Глобальный экземпляр компонентов
ui_components = UIComponents()
