from .scrappers import search_arxiv
from .scrappers import search_scholar, search_google
from .scrappers import search_openalex

from typing import List, Dict, Any

def collect_results(query: str, base: str, limit_per_source: int = 5) -> List[Dict[str, Any]]:
        
        all_papers: List[Dict[str, Any]] = []
        # 1. ArXiv
        if "arxiv" == base:
            try:
                for p in search_arxiv(query, max_results=limit_per_source):
                    all_papers.append({
                        "source": "arXiv",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("summary") or "Resumen no disponible",
                        "link": p.get("link", "N/A"),
                        "citations": "N/A"
                    })
            except Exception as e:
                print(f"⚠️ Error en ArXiv: {e}")

        # 2. Google Scholar
        if "google_scholar" == base:
            try:
                for p in search_scholar(query, num_results=limit_per_source):
                    all_papers.append({
                        "source": "Google Scholar",
                        "title": p.get("title", "Sin título"),
                    "summary": p.get("snippet") or "Resumen no disponible",
                    "link": p.get("link", "N/A"),
                    "citations": p.get("cited_by", 0)
                })
            except Exception as e:
                print(f"⚠️ Error en Google Scholar: {e}")

        # 3. OpenAlex
        if "openalex" == base:
            try:
                for p in search_openalex(query, per_page=limit_per_source):
                    all_papers.append({
                        "source": "OpenAlex",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("abstract") or "Resumen no disponible",
                        "link": p.get("doi", "N/A"),
                        "citations": p.get("citations", 0)
                    })
            except Exception as e:
                print(f"⚠️ Error en OpenAlex: {e}")
        
        # 4. Google Web Search
        if "google_web" == base:
            try:
                for p in search_google(query, num_results=limit_per_source):
                    all_papers.append({
                        "source": "Web",
                        "title": p.get("title", "Sin título"),
                        "summary": p.get("snippet") or "Resumen no disponible",
                        "link": p.get("url", "N/A"),
                        "citations": "N/A"
                    })
            except Exception as e:
                print(f"⚠️ Error en Google Web Search: {e}")

        return all_papers