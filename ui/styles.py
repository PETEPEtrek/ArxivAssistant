"""
Модуль стилей для ArXiv Assistant
"""

import streamlit as st

def apply_custom_styles():
    """
    Применяет кастомные CSS стили к приложению
    """
    st.markdown("""
    <style>
        .stButton > button {
            border-radius: 5px;
        }
        
        .article-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #fafafa;
            cursor: pointer;
        }
        
        .article-title {
            color: #333;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .article-authors {
            color: #666;
            margin-bottom: 5px;
            font-style: italic;
        }
        
        .article-abstract {
            color: #444;
            line-height: 1.4;
            margin-bottom: 10px;
        }
        
        .back-button {
            margin-bottom: 20px;
        }
        
        .chat-message-user {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }
        
        .chat-message-assistant {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
        }
        
        .search-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .main-container {
            padding: 2rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def get_article_card_style():
    """
    Возвращает стиль для карточки статьи
    """
    return """
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background-color: #fafafa;
    cursor: pointer;
    """

def get_chat_container_style():
    """
    Возвращает стиль для контейнера чата
    """
    return """
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
    """
