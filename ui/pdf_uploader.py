"""
Модуль для загрузки и обработки PDF файлов
"""

import streamlit as st
import os
import hashlib
from typing import Dict, Optional
import logging
from datetime import datetime
from paper_rag.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class PDFUploader:
    """
    Класс для загрузки и обработки PDF файлов
    """
    
    def __init__(self):
        """
        Инициализация загрузчика PDF
        """
        self.uploaded_files_dir = "uploaded_pdfs"
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """
        Создание директории для загруженных файлов
        """
        if not os.path.exists(self.uploaded_files_dir):
            os.makedirs(self.uploaded_files_dir)
            logger.info(f"Создана директория {self.uploaded_files_dir}")
    
    def generate_arxiv_id(self, filename: str, content: bytes) -> str:
        """
        Генерация уникального ID для загруженной статьи
        
        Args:
            filename: Имя файла
            content: Содержимое файла
            
        Returns:
            Уникальный ID в формате arXiv
        """
        content_hash = hashlib.md5(content).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"uploaded_{timestamp}_{content_hash}"
    
    def process_uploaded_pdf(self, uploaded_file) -> Optional[Dict]:
        """
        Обработка загруженного PDF файла
        
        Args:
            uploaded_file: Загруженный файл Streamlit
            
        Returns:
            Словарь с информацией о статье или None при ошибке
        """
        try:
            if uploaded_file is None:
                return None
            
            # Проверяем тип файла
            if not uploaded_file.name.lower().endswith('.pdf'):
                st.error("❌ Поддерживаются только PDF файлы")
                return None
            
            # Читаем содержимое файла
            content = uploaded_file.read()
            
            # Генерируем уникальный ID
            arxiv_id = self.generate_arxiv_id(uploaded_file.name, content)
            
            # Сохраняем файл
            file_path = os.path.join(self.uploaded_files_dir, f"{arxiv_id}.pdf")
            with open(file_path, "wb") as f:
                f.write(content)
            

            
            # Извлекаем метаданные и текст
            article_info = self._extract_article_info(file_path, arxiv_id, uploaded_file.name)
            
            if article_info:
                # Добавляем информацию о загруженном файле
                article_info['uploaded_file'] = True
                article_info['file_path'] = file_path
                article_info['original_filename'] = uploaded_file.name
                article_info['upload_timestamp'] = datetime.now().isoformat()
                
                return article_info
            else:
                # Удаляем файл если не удалось обработать
                os.remove(file_path)
                return None
                
        except Exception as e:
            logger.error(f"Ошибка обработки PDF: {e}")
            st.error(f"❌ Ошибка обработки PDF файла: {str(e)}")
            return None
    
    def _extract_article_info(self, file_path: str, arxiv_id: str, original_filename: str) -> Optional[Dict]:
        """
        Извлечение информации о статье из PDF
        
        Args:
            file_path: Путь к PDF файлу
            arxiv_id: Сгенерированный ID
            original_filename: Оригинальное имя файла
            
        Returns:
            Словарь с информацией о статье
        """
        try:
            
            # Используем PDF процессор
            pdf_processor = PDFProcessor()
            
            # Извлекаем текст используя PyPDF2
            extracted_data = pdf_processor.extract_text_pypdf2(file_path)
            if not extracted_data:
                logger.warning(f"Не удалось извлечь текст из {file_path}")
                return self._create_basic_article_info(file_path, arxiv_id, original_filename)
            
            # Получаем текст и метаданные
            text_content = extracted_data.get('text', '')
            metadata = extracted_data.get('metadata', {})
            
            # Создаем информацию о статье
            article_info = {
                'arxiv_id': arxiv_id,
                'title': metadata.get('title', original_filename.replace('.pdf', '')),
                'authors': metadata.get('authors', ['Неизвестный автор']),
                'abstract': self._extract_abstract(text_content),
                'full_abstract': self._extract_abstract(text_content),
                'published': metadata.get('published', datetime.now().isoformat()),
                'link': None,  # Для загруженных файлов нет ссылки
                'pdf_link': file_path,
                'uploaded_file': True,
                'file_path': file_path,
                'original_filename': original_filename,
                'text_content': text_content,
                'upload_timestamp': datetime.now().isoformat()
            }
            
            return article_info
            
        except Exception as e:
            logger.error(f"Ошибка извлечения информации из PDF: {e}")
            return self._create_basic_article_info(file_path, arxiv_id, original_filename)
    
    def _create_basic_article_info(self, file_path: str, arxiv_id: str, original_filename: str) -> Dict:
        """
        Создание базовой информации о статье если не удалось извлечь из PDF
        
        Args:
            file_path: Путь к PDF файлу
            arxiv_id: Сгенерированный ID
            original_filename: Оригинальное имя файла
            
        Returns:
            Базовая информация о статье
        """
        return {
            'arxiv_id': arxiv_id,
            'title': original_filename.replace('.pdf', ''),
            'authors': ['Неизвестный автор'],
            'abstract': f"PDF файл: {original_filename}",
            'full_abstract': f"PDF файл: {original_filename}",
            'published': datetime.now().isoformat(),
            'link': None,
            'pdf_link': file_path,
            'uploaded_file': True,
            'file_path': file_path,
            'original_filename': original_filename,
            'text_content': None,
            'upload_timestamp': datetime.now().isoformat()
        }
    
    def _extract_abstract(self, text_content: str) -> str:
        """
        Извлечение аннотации из текста
        
        Args:
            text_content: Текст статьи
            
        Returns:
            Аннотация или первые 200 символов
        """
        if not text_content:
            return "Аннотация недоступна"
        
        # Ищем ключевые слова для аннотации
        abstract_keywords = ['abstract', 'аннотация', 'резюме', 'summary']
        text_lower = text_content.lower()
        
        for keyword in abstract_keywords:
            if keyword in text_lower:
                # Ищем позицию ключевого слова
                pos = text_lower.find(keyword)
                # Берем текст после ключевого слова (до 500 символов)
                abstract_start = pos + len(keyword)
                abstract_text = text_content[abstract_start:abstract_start + 500].strip()
                
                # Очищаем текст
                lines = abstract_text.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10:  # Убираем короткие строки
                        clean_lines.append(line)
                
                if clean_lines:
                    return ' '.join(clean_lines[:3])  # Первые 3 строки
        
        # Fallback: первые 200 символов
        return text_content[:200] + "..." if len(text_content) > 200 else text_content
    
    def get_uploaded_articles(self) -> list:
        """
        Получение списка загруженных статей
        
        Returns:
            Список загруженных статей
        """
        uploaded_articles = []
        
        try:
            if not os.path.exists(self.uploaded_files_dir):
                return uploaded_articles
            
            for filename in os.listdir(self.uploaded_files_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(self.uploaded_files_dir, filename)
                    
                    # Извлекаем arxiv_id из имени файла
                    arxiv_id = filename.replace('.pdf', '')
                    
                    # Создаем базовую информацию
                    article_info = {
                        'arxiv_id': arxiv_id,
                        'title': filename.replace('.pdf', ''),
                        'authors': ['Загруженный файл'],
                        'abstract': f"PDF файл: {filename}",
                        'full_abstract': f"PDF файл: {filename}",
                        'published': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        'link': None,
                        'pdf_link': file_path,
                        'uploaded_file': True,
                        'file_path': file_path,
                        'original_filename': filename,
                        'text_content': None,
                        'upload_timestamp': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    }
                    
                    uploaded_articles.append(article_info)
                    
        except Exception as e:
            logger.error(f"Ошибка получения списка загруженных статей: {e}")
        
        return uploaded_articles
    
    def delete_uploaded_article(self, arxiv_id: str) -> bool:
        """
        Удаление загруженной статьи
        
        Args:
            arxiv_id: ID статьи для удаления
            
        Returns:
            True если удаление успешно
        """
        try:
            file_path = os.path.join(self.uploaded_files_dir, f"{arxiv_id}.pdf")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Загруженная статья удалена: {file_path}")
                return True
            else:
                logger.warning(f"Файл не найден: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка удаления статьи: {e}")
            return False

# Глобальный экземпляр загрузчика
pdf_uploader = PDFUploader()
