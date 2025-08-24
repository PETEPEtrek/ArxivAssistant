"""
Модуль для обработки PDF статей из arXiv
"""

import re
from typing import Dict, Optional
import requests
from pathlib import Path
import logging
import PyPDF2

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Класс для обработки PDF файлов статей из arXiv
    """
    
    def __init__(self, data_dir: str = "paper_rag/data/papers"):
        """
        Инициализация процессора PDF
        
        Args:
            data_dir: Директория для сохранения PDF файлов
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_pdf(self, arxiv_id: str, pdf_url: str) -> Optional[str]:
        """
        Скачивание PDF файла статьи
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            pdf_url: URL для скачивания PDF
            
        Returns:
            Путь к скачанному файлу или None при ошибке
        """
        try:
            # Очищаем arxiv_id от лишних символов
            clean_id = re.sub(r'[^\w\.-]', '_', arxiv_id)
            pdf_path = self.data_dir / f"{clean_id}.pdf"
            
            # Проверяем, не скачан ли уже файл
            if pdf_path.exists():
                logger.info(f"PDF уже существует: {pdf_path}")
                return str(pdf_path)
            
            # Скачиваем PDF
            logger.info(f"Скачивание PDF: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Сохраняем файл
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF сохранен: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Ошибка при скачивании PDF {arxiv_id}: {e}")
            return None
    
    def extract_text_pypdf2(self, pdf_path: str) -> Optional[Dict]:
        """
        Резервный метод извлечения текста с помощью PyPDF2
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Словарь с извлеченным текстом и метаданными
        """
        if PyPDF2 is None:
            logger.error("PyPDF2 не установлен")
            return None
        
        try:
            logger.info(f"Извлечение текста с PyPDF2: {pdf_path}")
            
            text_by_page = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        text_by_page.append({
                            'page': page_num,
                            'text': page_text
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка извлечения страницы {page_num}: {e}")
                        continue
            
            # Объединяем весь текст
            full_text = '\n\n'.join([page['text'] for page in text_by_page])
            
            extracted_data = {
                'text': full_text,
                'pages': text_by_page,
                'metadata': {
                    'source': 'pypdf2',
                    'file_path': pdf_path,
                    'extraction_method': 'pypdf2',
                    'total_pages': len(text_by_page)
                }
            }
            
            logger.info(f"Успешно извлечен текст: {len(full_text)} символов, {len(text_by_page)} страниц")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста с PyPDF2: {e}")
            return None

    
    def process_article(self, arxiv_id: str, pdf_url: str) -> Optional[Dict]:
        """
        Полный цикл обработки статьи: скачивание и извлечение текста
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            pdf_url: URL для скачивания PDF
            
        Returns:
            Словарь с извлеченным текстом и метаданными
        """
        logger.info(f"Начало обработки статьи: {arxiv_id}")
        
        # Скачиваем PDF
        pdf_path = self.download_pdf(arxiv_id, pdf_url)
        if not pdf_path:
            return None
        
        # Извлекаем текст (пробуем все доступные методы)
        extracted_data = self.extract_text_pypdf2(pdf_path)
        if not extracted_data:
            logger.error(f"Не удалось извлечь текст из {arxiv_id}")
            return None
        
        # Добавляем информацию о статье
        extracted_data['metadata']['arxiv_id'] = arxiv_id
        extracted_data['metadata']['pdf_url'] = pdf_url
        extracted_data['metadata']['pdf_path'] = pdf_path  # Добавляем путь к PDF для визуального анализа
        
        logger.info(f"Обработка статьи {arxiv_id} завершена успешно")
        return extracted_data
    
    def clean_text(self, text: str) -> str:
        """
        Очистка извлеченного текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        # Удаляем лишние пробелы и переносы
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем странные символы
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\/\[\]\{\}\"\']+', ' ', text)
        
        # Удаляем повторяющиеся знаки препинания
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()

# Глобальный экземпляр процессора
pdf_processor = PDFProcessor()
