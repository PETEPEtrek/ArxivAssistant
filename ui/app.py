"""
Главное приложение ArXiv Assistant
"""

import streamlit as st
from .arxiv_api import arxiv_api
from .chat import chat_manager
from .components import ui_components
from .styles import apply_custom_styles

# Конфигурация страницы
st.set_page_config(
    page_title="ArXiv Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class ArxivAssistantApp:
    """
    Главный класс приложения ArXiv Assistant
    """
    
    def __init__(self):
        self.initialize_session_state()
        apply_custom_styles()
    
    def initialize_session_state(self):
        """
        Инициализация состояния сессии
        """
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "search"
    
    def run(self):
        """
        Запуск приложения
        """
        # Маршрутизация между страницами
        if st.session_state.current_page == "search":
            self.search_page()
        elif st.session_state.current_page == "article_view":
            self.article_view_page()
    
    def search_page(self):
        """
        Главная страница поиска
        """
        # Заголовок приложения
        ui_components.display_page_header()
        
        # Секция загрузки PDF
        uploaded_article = ui_components.display_pdf_upload_section()
        
        # Если PDF загружен и обработан, переходим к статье
        if uploaded_article:
            st.session_state.current_page = "article_view"
            st.session_state.selected_article = uploaded_article
            
            # Запускаем асинхронную обработку статьи для RAG
            ui_components._queue_article_processing(uploaded_article)
            
            st.rerun()
        
        # Секция с уже загруженными статьями
        selected_uploaded = ui_components.display_uploaded_articles_section()
        
        # Если выбрана загруженная статья, переходим к ней
        if selected_uploaded:
            st.session_state.current_page = "article_view"
            st.session_state.selected_article = selected_uploaded
            
            # Запускаем асинхронную обработку статьи для RAG
            ui_components._queue_article_processing(selected_uploaded)
            
            st.rerun()
        
        # Форма поиска
        search_query, max_results = ui_components.display_search_form()
        
        # Выполнение поиска
        if search_query:
            with st.spinner("🔍 Поиск статей..."):
                articles = arxiv_api.search_articles(search_query, max_results)
            
            # Отображение результатов
            ui_components.display_search_results(articles)
        
        # Информация о приложении
        #ui_components.display_app_info()
    
    def article_view_page(self):
        """
        Страница просмотра отдельной статьи с возможностью диалога
        """
        if 'selected_article' not in st.session_state:
            st.error("Статья не выбрана. Возвращение к поиску...")
            st.session_state.current_page = "search"
            st.rerun()
            return
        
        article = st.session_state.selected_article
        
        # Заголовок и информация о статье + кнопка обновления
        ui_components.display_article_header(article)
        
        # Кнопка обновления страницы
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Обновить страницу", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🏠 На главную", use_container_width=True):
                # Очищаем диалог при выходе со страницы статьи
                arxiv_id = article.get('arxiv_id')
                if arxiv_id:
                    chat_manager.clear_chat(arxiv_id)
                
                st.session_state.current_page = "search"
                st.rerun()
        
        # Аннотация статьи
        ui_components.display_article_abstract(article)
        
        # Секция суммаризации
        ui_components.display_summarize_section(article)
        
        # Индикатор статуса RAG обработки
        ui_components.display_rag_status_indicator(article)
        
        # Селектор LLM модели
        selected_model = ui_components.display_llm_model_selector()
        
        # Если выбрана новая модель, переключаемся на неё
        if selected_model and selected_model != st.session_state.get('selected_llm_model'):
            if chat_manager.switch_model(selected_model):
                st.success(f"✅ Переключение на модель: {selected_model['type']} {selected_model['name']}")
                st.rerun()
            else:
                st.error(f"❌ Не удалось переключиться на модель: {selected_model['type']} {selected_model['name']}")
        
        # Секция диалога
        self.display_chat_section(article)
    
    def display_chat_section(self, article):
        """
        Отображение секции чата
        
        Args:
            article: Информация о статье
        """
        st.markdown("---")
        st.markdown("### 💬 Обсуждение статьи")
        
        # Отображение истории чата
        arxiv_id = article.get('arxiv_id')
        chat_manager.display_chat_history(arxiv_id)
        
        # Расширенное поле ввода с подсказками
        user_input = ui_components.display_enhanced_chat_input(article)
        
        # Обработка нового сообщения
        if user_input:
            self.process_chat_message(user_input, article)
    
    def process_chat_message(self, user_input: str, article: dict):
        """
        Обработка нового сообщения в чате
        
        Args:
            user_input: Сообщение пользователя
            article: Информация о статье
        """
        arxiv_id = article.get('arxiv_id')
        
        # Добавляем сообщение пользователя
        chat_manager.add_message('user', user_input, arxiv_id)
        
        # Генерируем и добавляем ответ ассистента
        response = chat_manager.generate_response(user_input, article)
        chat_manager.add_message('assistant', response, arxiv_id)
        
        # Перезагружаем страницу для отображения новых сообщений
        st.rerun()

def main():
    """
    Точка входа в приложение
    """
    app = ArxivAssistantApp()
    app.run()

if __name__ == "__main__":
    main()
