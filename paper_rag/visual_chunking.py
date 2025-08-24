"""
Модуль для чанкинга текста по визуальным заголовкам (размер шрифта, жирность)
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

import fitz
import pdfplumber

logger = logging.getLogger(__name__)

class VisualHeaderDetector:
    """
    Класс для определения заголовков по визуальному форматированию
    """
    
    def __init__(self):
        """
        Инициализация детектора визуальных заголовков
        """
        self.min_font_size_ratio = 1.1  # Заголовок должен быть минимум на 10% больше обычного текста
        self.min_header_length = 3
        self.max_header_length = 100
        
    def extract_visual_headers(self, pdf_path: str) -> List[Dict]:
        """
        Извлечение заголовков на основе визуального форматирования
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список заголовков с информацией о форматировании
        """
        logger.info(f"Анализ визуального форматирования PDF: {pdf_path}")
        
        # Пробуем разные методы
        headers = []
        
        if fitz:
            headers = self._extract_headers_with_pymupdf(pdf_path)
        elif pdfplumber:
            headers = self._extract_headers_with_pdfplumber(pdf_path)
        else:
            logger.warning("Библиотеки для визуального анализа PDF недоступны")
            return []
        
        # Фильтруем и валидируем заголовки
        valid_headers = self._filter_and_validate_headers(headers)
        
        logger.info(f"Найдено {len(valid_headers)} визуальных заголовков")
        return valid_headers
    
    def _extract_headers_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """
        Извлечение заголовков с помощью PyMuPDF
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список потенциальных заголовков
        """
        try:
            doc = fitz.open(pdf_path)
            headers = []
            
            # Сначала определяем средний размер шрифта в документе
            font_sizes = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                font_size = span.get("size", 0)
                                if font_size > 0:
                                    font_sizes.append(font_size)
            
            if not font_sizes:
                logger.warning("Не найдена информация о размерах шрифтов")
                doc.close()
                return []
            
            # Определяем средний и максимальный размер шрифта
            avg_font_size = sum(font_sizes) / len(font_sizes)
            max_font_size = max(font_sizes)
            header_threshold = avg_font_size * self.min_font_size_ratio
            
            logger.info(f"Средний размер шрифта: {avg_font_size:.1f}, порог для заголовков: {header_threshold:.1f}")
            
            # Извлекаем заголовки
            char_position = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            line_font_size = 0
                            line_is_bold = False
                            line_y = line.get("bbox", [0, 0, 0, 0])[1]  # Y координата
                            
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                font_size = span.get("size", 0)
                                font_flags = span.get("flags", 0)
                                
                                # Проверяем жирность (флаг 16 = bold)
                                is_bold = bool(font_flags & 16)
                                
                                if text:
                                    line_text += text + " "
                                    line_font_size = max(line_font_size, font_size)
                                    line_is_bold = line_is_bold or is_bold
                            
                            line_text = line_text.strip()
                            
                            # Проверяем, подходит ли строка для заголовка
                            # Приоритет жирности над размером шрифта
                            if (line_text and 
                                len(line_text) >= self.min_header_length and
                                len(line_text) <= self.max_header_length and
                                (line_is_bold or line_font_size >= header_threshold)):
                                
                                headers.append({
                                    'text': line_text,
                                    'font_size': line_font_size,
                                    'is_bold': line_is_bold,
                                    'page': page_num + 1,
                                    'y_position': line_y,
                                    'char_position': char_position,
                                    'score': self._calculate_header_score(
                                        line_text, line_font_size, line_is_bold, 
                                        avg_font_size, max_font_size
                                    )
                                })
                            
                            char_position += len(line_text) + 1
            
            doc.close()
            return headers
            
        except Exception as e:
            logger.error(f"Ошибка анализа PDF с PyMuPDF: {e}")
            return []
    
    def _extract_headers_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """
        Извлечение заголовков с помощью pdfplumber
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список потенциальных заголовков
        """
        try:
            headers = []
            char_position = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                # Собираем информацию о размерах шрифтов
                font_sizes = []
                
                for page in pdf.pages:
                    chars = page.chars
                    for char in chars:
                        size = char.get('size', 0)
                        if size > 0:
                            font_sizes.append(size)
                
                if not font_sizes:
                    logger.warning("Не найдена информация о размерах шрифтов")
                    return []
                
                avg_font_size = sum(font_sizes) / len(font_sizes)
                header_threshold = avg_font_size * self.min_font_size_ratio
                
                logger.info(f"Средний размер шрифта: {avg_font_size:.1f}, порог для заголовков: {header_threshold:.1f}")
                
                # Извлекаем потенциальные заголовки
                for page_num, page in enumerate(pdf.pages):
                    # Группируем символы по строкам
                    lines = self._group_chars_to_lines(page.chars)
                    
                    for line in lines:
                        line_text = line['text'].strip()
                        line_font_size = line['max_font_size']
                        
                        # Простая эвристика для определения жирности
                        # (если большинство символов имеют одинаковый шрифт)
                        line_is_bold = self._estimate_boldness(line['chars'])
                        
                        if (line_text and 
                            len(line_text) >= self.min_header_length and
                            len(line_text) <= self.max_header_length and
                            (line_font_size >= header_threshold or line_is_bold)):
                            
                            headers.append({
                                'text': line_text,
                                'font_size': line_font_size,
                                'is_bold': line_is_bold,
                                'page': page_num + 1,
                                'y_position': line['y_position'],
                                'char_position': char_position,
                                'score': self._calculate_header_score(
                                    line_text, line_font_size, line_is_bold,
                                    avg_font_size, max(font_sizes)
                                )
                            })
                        
                        char_position += len(line_text) + 1
            
            return headers
            
        except Exception as e:
            logger.error(f"Ошибка анализа PDF с pdfplumber: {e}")
            return []
    
    def _group_chars_to_lines(self, chars: List[Dict]) -> List[Dict]:
        """
        Группировка символов в строки
        
        Args:
            chars: Список символов с их характеристиками
            
        Returns:
            Список строк с агрегированной информацией
        """
        if not chars:
            return []
        
        lines = []
        current_line = None
        tolerance = 2  # Допуск для Y координаты
        
        for char in sorted(chars, key=lambda x: (x.get('y0', 0), x.get('x0', 0))):
            y_pos = char.get('y0', 0)
            
            if current_line is None or abs(y_pos - current_line['y_position']) > tolerance:
                # Начинаем новую строку
                if current_line:
                    lines.append(current_line)
                
                current_line = {
                    'text': char.get('text', ''),
                    'y_position': y_pos,
                    'max_font_size': char.get('size', 0),
                    'chars': [char]
                }
            else:
                # Добавляем к текущей строке
                current_line['text'] += char.get('text', '')
                current_line['max_font_size'] = max(
                    current_line['max_font_size'], 
                    char.get('size', 0)
                )
                current_line['chars'].append(char)
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _estimate_boldness(self, chars: List[Dict]) -> bool:
        """
        Оценка жирности текста на основе имен шрифтов
        
        Args:
            chars: Список символов
            
        Returns:
            True если текст вероятно жирный
        """
        bold_indicators = ['bold', 'Bold', 'BOLD', 'Heavy', 'Black']
        
        for char in chars:
            font_name = char.get('fontname', '').lower()
            if any(indicator.lower() in font_name for indicator in bold_indicators):
                return True
        
        return False
    
    def _calculate_header_score(self, text: str, font_size: float, is_bold: bool,
                               avg_font_size: float, max_font_size: float) -> float:
        """
        Вычисление score для потенциального заголовка
        
        Args:
            text: Текст заголовка
            font_size: Размер шрифта
            is_bold: Жирный ли текст
            avg_font_size: Средний размер шрифта в документе
            max_font_size: Максимальный размер шрифта
            
        Returns:
            Score заголовка (чем больше, тем лучше)
        """
        score = 0.0
        
        # Баллы за размер шрифта
        if font_size > avg_font_size:
            size_ratio = font_size / avg_font_size
            score += min(size_ratio, 3.0)  # Максимум 3 балла
        
        # Баллы за жирность
        if is_bold:
            score += 1.0
        
        # Баллы за длину (оптимальная длина заголовка)
        if 5 <= len(text) <= 50:
            score += 0.5
        
        # Штраф за слишком длинный текст
        if len(text) > 80:
            score -= 1.0
        
        # Бонус если текст начинается с заглавной буквы
        if text and text[0].isupper():
            score += 0.3
        
        # Бонус за нумерацию (1., 2., I., etc.)
        if re.match(r'^[\d\w]+[\.\)]\s*[A-Z]', text):
            score += 0.5
        
        return score
    
    def _filter_and_validate_headers(self, headers: List[Dict]) -> List[Dict]:
        """
        Фильтрация и валидация найденных заголовков
        
        Args:
            headers: Список потенциальных заголовков
            
        Returns:
            Отфильтрованный список заголовков
        """
        if not headers:
            return []
        
        # Сортируем по score (убывание)
        headers.sort(key=lambda x: x['score'], reverse=True)
        
        # Удаляем дубликаты и близкие заголовки
        filtered_headers = []
        
        for header in headers:
            text = header['text'].strip()
            
            # СТРОГАЯ валидация заголовка
            if not self._is_valid_header_text(text):
                logger.debug(f"Отклонен заголовок '{text}' - не прошел валидацию")
                continue
            
            # Проверяем на дубликаты
            is_duplicate = False
            for existing in filtered_headers:
                if (abs(header['char_position'] - existing['char_position']) < 100 and
                    self._texts_similar(text, existing['text'])):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_headers.append(header)
                logger.debug(f"Добавлен заголовок: '{text}' (score: {header['score']:.2f})")
        
        # Сортируем по позиции в документе
        filtered_headers.sort(key=lambda x: x['char_position'])
        
        logger.info(f"После фильтрации осталось {len(filtered_headers)} заголовков из {len(headers)}")
        return filtered_headers
    
    def _is_valid_header_text(self, text: str) -> bool:
        """
        Проверка валидности текста заголовка по строгим критериям:
        1. Начинается с цифры (1 Introduction, 2.1 Methods)
        2. Начинается с буквы + пробел (E Additional Details)
        3. Стандартные разделы (Abstract, Acknowledgments, References)
        
        Args:
            text: Текст для проверки
            
        Returns:
            True если текст может быть заголовком
        """
        text = text.strip()
        
        # Минимальная длина
        if len(text) < self.min_header_length:
            return False
        
        # Максимальная длина
        if len(text) > self.max_header_length:
            return False
        
        # Исключаем URL и email
        if '@' in text or 'http' in text.lower():
            return False
        
        # Исключаем чисто математические выражения
        math_symbols = ['=', '≤', '≥', '±', '∞', '∑', '∏', '∫']
        if any(symbol in text for symbol in math_symbols):
            return False
        
        # ДОПОЛНИТЕЛЬНЫЕ СТРОГИЕ КРИТЕРИИ:
        
        # Исключаем текст, который заканчивается на "but", "and", "or" (незавершенные предложения)
        if text.rstrip().endswith(('but', 'and', 'or', 'the', 'a', 'an')):
            return False
        
        # Исключаем текст с множественными пробелами в середине (плохо отформатированный)
        if '  ' in text:
            return False
        
        # Исключаем текст, который выглядит как часть предложения
        if text.count(' ') > 8:  # Слишком много слов для заголовка
            return False
        
        # СТРОГИЕ КРИТЕРИИ:
        
        # 1. Начинается с цифры (нумерованные разделы)
        if re.match(r'^\d+', text):
            # Должен иметь текст после числа (с точкой или без)
            if re.match(r'^\d+(\.\d+)*[\.\s]*[A-Za-z]', text):
                return True
        
        # 2. Начинается с одной буквы + пробел (приложения)
        if re.match(r'^[A-Z]\s+[A-Za-z]', text):
            return True
        
        # 3. Стандартные разделы статьи
        standard_sections = [
            'Abstract', 'ABSTRACT',
            'Introduction', 'INTRODUCTION', 
            'Related Work', 'RELATED WORK',
            'Background', 'BACKGROUND',
            'Methods', 'METHODS', 'Methodology', 'METHODOLOGY',
            'Results', 'RESULTS',
            'Discussion', 'DISCUSSION',
            'Conclusion', 'CONCLUSION', 'Conclusions', 'CONCLUSIONS',
            'Acknowledgments', 'ACKNOWLEDGMENTS', 'Acknowledgements', 'ACKNOWLEDGEMENTS',
            'References', 'REFERENCES',
            'Bibliography', 'BIBLIOGRAPHY',
            'Appendix', 'APPENDIX'
        ]
        
        # Точное совпадение со стандартными разделами
        if text in standard_sections:
            return True
        
        # Appendix с буквой (Appendix A, Appendix B)
        if re.match(r'^Appendix\s+[A-Z]', text, re.IGNORECASE):
            return True
        
        # Все остальное отклоняем
        return False
    
    def _texts_similar(self, text1: str, text2: str) -> bool:
        """
        Проверка схожести двух текстов
        
        Args:
            text1, text2: Тексты для сравнения
            
        Returns:
            True если тексты схожи
        """
        # Простая проверка на схожесть
        text1_clean = re.sub(r'\W+', '', text1.lower())
        text2_clean = re.sub(r'\W+', '', text2.lower())
        
        if text1_clean == text2_clean:
            return True
        
        # Проверяем пересечение слов
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1.intersection(words2)) / max(len(words1), len(words2)) > 0.7:
            return True
        
        return False

# Глобальный экземпляр детектора
visual_header_detector = VisualHeaderDetector()
