"""
Модуль для работы с ArXiv API
"""

import requests
import feedparser
import re
import html
from typing import List, Dict
import streamlit as st

class ArxivAPI:
    """
    Класс для работы с ArXiv API
    """
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"
    
    def search_articles(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Поиск статей в arXiv API
        
        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            
        Returns:
            Список словарей с информацией о статьях
        """
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # Формируем URL запроса
        url = self.base_url + "&".join([f"{k}={v}" for k, v in params.items()])
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            articles = []
            for entry in feed.entries:
                article = self._parse_entry(entry)
                articles.append(article)
            
            return articles
        
        except requests.RequestException as e:
            st.error(f"Ошибка при запросе к arXiv API: {e}")
            return []
        except Exception as e:
            st.error(f"Неожиданная ошибка: {e}")
            return []
    
    def _parse_entry(self, entry) -> Dict:
        """
        Парсинг отдельной записи из ответа API
        
        Args:
            entry: Запись из feedparser
            
        Returns:
            Словарь с информацией о статье
        """
        # Извлекаем авторов
        authors = self._extract_authors(entry)
        
        # Извлекаем и очищаем аннотацию
        abstract, full_abstract = self._extract_abstract(entry)
        
        # Извлекаем ссылку на PDF
        pdf_link = self._extract_pdf_link(entry)
        
        # Очищаем заголовок от HTML тегов и лишних символов
        title = self._clean_title(entry)
        
        return {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'full_abstract': full_abstract,
            'link': entry.link if hasattr(entry, 'link') else "",
            'pdf_link': pdf_link,
            'published': entry.published if hasattr(entry, 'published') else "",
            'arxiv_id': entry.id.split('/')[-1] if hasattr(entry, 'id') else ""
        }
    
    def _extract_authors(self, entry) -> List[str]:
        """
        Извлечение списка авторов из записи
        """
        authors = []
        if hasattr(entry, 'authors'):
            authors = [author.name for author in entry.authors]
        elif hasattr(entry, 'author'):
            authors = [entry.author]
        return authors
    
    def _extract_abstract(self, entry) -> tuple:
        """
        Извлечение аннотации из записи
        
        Returns:
            Кортеж (краткая аннотация, полная аннотация)
        """
        abstract = ""
        full_abstract = ""
        if hasattr(entry, 'summary'):
            full_abstract = re.sub(r'\s+', ' ', entry.summary.strip())
            abstract = full_abstract[:300] + "..." if len(full_abstract) > 300 else full_abstract
        return abstract, full_abstract
    
    def _extract_pdf_link(self, entry) -> str:
        """
        Извлечение ссылки на PDF из записи
        """
        pdf_link = ""
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.type == 'application/pdf':
                    pdf_link = link.href
                    break
        return pdf_link
    
    def get_source_links(self, arxiv_id: str) -> Dict[str, str]:
        """
        Получение ссылок на различные форматы исходного кода
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            
        Returns:
            Словарь с ссылками на различные форматы
        """
        # Формируем базовый URL для статьи
        base_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        # Возможные форматы исходного кода
        source_links = {
            'source': base_url,  # Основной исходный код (обычно tar.gz)
            'latex': f"https://arxiv.org/format/{arxiv_id}/source",  # LaTeX исходный код
            'pdf': f"https://arxiv.org/pdf/{arxiv_id}.pdf",  # PDF версия
            'abs': f"https://arxiv.org/abs/{arxiv_id}"  # Страница аннотации
        }
        
        return source_links
    
    def download_source(self, arxiv_id: str, format_type: str = 'source') -> str:
        """
        Скачивание исходного кода статьи
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            format_type: Тип формата ('source', 'latex', 'pdf')
            
        Returns:
            Путь к скачанному файлу или None при ошибке
        """
        import os
        from pathlib import Path
        
        try:
            source_links = self.get_source_links(arxiv_id)
            
            if format_type not in source_links:
                st.error(f"Неизвестный формат: {format_type}")
                return None
            
            url = source_links[format_type]
            
            # Создаем папку для исходного кода
            source_dir = Path("paper_rag/data/sources")
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Определяем имя файла
            if format_type == 'source':
                filename = f"{arxiv_id}_source.tar.gz"
            elif format_type == 'latex':
                filename = f"{arxiv_id}_latex.tar.gz"
            else:
                filename = f"{arxiv_id}_{format_type}"
            
            file_path = source_dir / filename
            
            # Скачиваем файл
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            st.success(f"Исходный код скачан: {file_path}")
            return str(file_path)
            
        except Exception as e:
            st.error(f"Ошибка при скачивании исходного кода: {e}")
            return None
    
    def get_available_formats(self, arxiv_id: str) -> List[str]:
        """
        Проверка доступных форматов для статьи
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            
        Returns:
            Список доступных форматов
        """
        available_formats = []
        
        try:
            # Проверяем PDF
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.head(pdf_url, timeout=5)
            if response.status_code == 200:
                available_formats.append('pdf')
            
            # Проверяем исходный код
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
            response = requests.head(source_url, timeout=5)
            if response.status_code == 200:
                available_formats.append('source')
            
            # Проверяем LaTeX формат
            latex_url = f"https://arxiv.org/format/{arxiv_id}/source"
            response = requests.head(latex_url, timeout=5)
            if response.status_code == 200:
                available_formats.append('latex')
                
        except Exception as e:
            st.warning(f"Ошибка при проверке форматов: {e}")
        
        return available_formats
    
    def _clean_title(self, entry) -> str:
        """
        Очистка заголовка от HTML тегов и лишних символов
        """
        title = entry.title if hasattr(entry, 'title') else "Без названия"
        title = html.unescape(title)  # Декодируем HTML entities
        title = re.sub(r'\s+', ' ', title.strip())  # Убираем лишние пробелы
        return title

# Глобальный экземпляр API
arxiv_api = ArxivAPI()
