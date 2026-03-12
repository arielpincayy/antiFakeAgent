from .arxiv import search_arxiv
from .googlescholar import search_scholar
from .openalex import search_openalex

# Esto permite importar directamente desde 'scrappers'
__all__ = ["search_arxiv", "search_scholar", "search_openalex"]