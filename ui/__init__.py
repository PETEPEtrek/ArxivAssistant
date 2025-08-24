"""
UI пакет для ArXiv Assistant
"""

from .arxiv_api import arxiv_api
from .chat import chat_manager
from .components import ui_components
from .styles import apply_custom_styles

__all__ = [
    'arxiv_api',
    'chat_manager', 
    'ui_components',
    'apply_custom_styles'
]
