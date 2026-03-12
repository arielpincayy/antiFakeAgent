from .scrappers import search_arxiv
from .scrappers import search_scholar
from .scrappers import search_openalex

from typing import List, Dict, Any

class Scrapping:
    def __init__(self):
        self.all_papers = []

    def collect_results(self, query: str, limit_per_source: int = 5) -> List[Dict[str, Any]]:
            # 1. ArXiv
            try:
                for p in search_arxiv(query, max_results=limit_per_source):
                    self.all_papers.append({
                        "source": "arXiv",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("summary") or "Resumen no disponible",
                        "link": p.get("link", "N/A"),
                        "citations": "N/A"
                    })
            except Exception as e:
                print(f"⚠️ Error en ArXiv: {e}")
    
            # 2. Google Scholar
            try:
                for p in search_scholar(query, num_results=limit_per_source):
                    self.all_papers.append({
                        "source": "Google Scholar",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("snippet") or "Resumen no disponible",
                        "link": p.get("link", "N/A"),
                        "citations": p.get("cited_by", 0)
                    })
            except Exception as e:
                print(f"⚠️ Error en Google Scholar: {e}")
    
            # 3. OpenAlex
            try:
                for p in search_openalex(query, per_page=limit_per_source):
                    self.all_papers.append({
                        "source": "OpenAlex",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("abstract") or "Resumen no disponible",
                        "link": p.get("doi", "N/A"),
                        "citations": p.get("citations", 0)
                    })
            except Exception as e:
                print(f"⚠️ Error en OpenAlex: {e}")
    
            return self.all_papers
    
    def clear_results(self):
        self.all_papers = []